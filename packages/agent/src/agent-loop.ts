/**
 * Agent loop that works with AgentMessage throughout.
 * Transforms to Message[] only at the LLM call boundary.
 */

import {
	type AssistantMessage,
	type Context,
	EventStream,
	streamSimple,
	type ToolResultMessage,
	validateToolArguments,
} from "@mariozechner/pi-ai";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentToolCall,
	AgentToolResult,
	DispatchRequest,
	DispatchResult,
	DispatchStatus,
	StreamFn,
} from "./types.js";

export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;

const MAX_TOOL_EXECUTION_ATTEMPTS = 10;
const MAX_STEP_REPEAT_ATTEMPTS = 10;
const DEFAULT_DISPATCH_MAX_DEPTH = 8;

type LoopCompletion = {
	status: DispatchStatus;
	reason?: string;
	toolResults: ToolResultMessage[];
};

type LoopRunOptions = {
	rootContext?: AgentContext;
	remainingDispatchDepth?: number;
	maxSteps?: number;
};

type LoopRuntime = {
	rootContext: AgentContext;
	remainingDispatchDepth: number;
	streamFn?: StreamFn;
};

/**
 * Start an agent loop with a new prompt message.
 * The prompt is added to the context and events are emitted for it.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	void runAgentLoop(
		prompts,
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

/**
 * Continue an agent loop from the current context without adding a new message.
 * Used for retries - context already has user message or tool results.
 *
 * **Important:** The last message in context must convert to a `user` or `toolResult` message
 * via `convertToLlm`. If it doesn't, the LLM provider will reject the request.
 * This cannot be validated here since `convertToLlm` is only called once per turn.
 */
export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const stream = createAgentStream();

	void runAgentLoopContinue(
		context,
		config,
		async (event) => {
			stream.push(event);
		},
		signal,
		streamFn,
	).then((messages) => {
		stream.end(messages);
	});

	return stream;
}

export async function runAgentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	const newMessages: AgentMessage[] = [...prompts];
	const currentContext: AgentContext = {
		...context,
		messages: [...context.messages, ...prompts],
	};

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });
	for (const prompt of prompts) {
		await emit({ type: "message_start", message: prompt });
		await emit({ type: "message_end", message: prompt });
	}

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn, {
		rootContext: shallowCloneAgentContext(context),
	});
	return newMessages;
}

export async function runAgentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	emit: AgentEventSink,
	signal?: AbortSignal,
	streamFn?: StreamFn,
): Promise<AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	if (context.messages[context.messages.length - 1].role === "assistant") {
		throw new Error("Cannot continue from message role: assistant");
	}

	const newMessages: AgentMessage[] = [];
	const currentContext: AgentContext = { ...context };

	await emit({ type: "agent_start" });
	await emit({ type: "turn_start" });

	await runLoop(currentContext, newMessages, config, signal, emit, streamFn, {
		rootContext: shallowCloneAgentContext(context),
	});
	return newMessages;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic shared by agentLoop and agentLoopContinue.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
	options: LoopRunOptions = {},
): Promise<LoopCompletion> {
	let firstTurn = true;
	// Check for steering messages at start (user may have typed while waiting)
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];
	let stepRepeatAttempts = 0;
	let stepsUsed = 0;
	const producedToolOutcomes: ToolResultOutcome[] = [];
	const runtime: LoopRuntime = {
		rootContext: options.rootContext ?? shallowCloneAgentContext(currentContext),
		remainingDispatchDepth: options.remainingDispatchDepth ?? DEFAULT_DISPATCH_MAX_DEPTH,
		streamFn,
	};

	// Outer loop: continues when queued follow-up messages arrive after agent would stop
	while (true) {
		let hasMoreToolCalls = true;

		// Inner loop: process tool calls and steering messages
		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				await emit({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			// Process pending messages (inject before next assistant response)
			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					await emit({ type: "message_start", message });
					await emit({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			// Stream assistant response
			if (options.maxSteps !== undefined && stepsUsed >= options.maxSteps) {
				await emit({ type: "agent_end", messages: newMessages });
				return { status: "limited", reason: "maxSteps", toolResults: getToolResultMessages(producedToolOutcomes) };
			}
			const stepContextStart = currentContext.messages.length;
			const stepNewMessagesStart = newMessages.length;
			const message = await streamAssistantResponse(currentContext, config, signal, emit, streamFn);
			newMessages.push(message);
			stepsUsed++;

			if (message.stopReason === "error" || message.stopReason === "aborted") {
				await emit({ type: "message_end", message });
				await emit({ type: "turn_end", message, toolResults: [] });
				await emit({ type: "agent_end", messages: newMessages });
				return {
					status: message.stopReason === "aborted" ? "aborted" : "failure",
					reason: getAssistantErrorMessage(message),
					toolResults: getToolResultMessages(producedToolOutcomes),
				};
			}

			// Check for tool calls
			const toolCalls = message.content.filter((c) => c.type === "toolCall");

			hasMoreToolCalls = false;
			if (toolCalls.length > 0) {
				const executedToolBatch = await executeToolCalls(
					currentContext,
					message,
					config,
					signal,
					emit,
					stepRepeatAttempts,
					runtime,
				);
				if (executedToolBatch.retry === "step") {
					stepRepeatAttempts++;
					currentContext.messages.splice(stepContextStart);
					newMessages.splice(stepNewMessagesStart);
					pendingMessages = executedToolBatch.injectMessages;
					firstTurn = true;
					continue;
				}

				producedToolOutcomes.push(...executedToolBatch.outcomes);
				hasMoreToolCalls = !executedToolBatch.terminate;

				await emit({ type: "message_end", message });
				const toolResults = getToolResultMessages(executedToolBatch.outcomes);
				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
					await emitToolResultMessage(result, emit);
				}
				await emit({ type: "turn_end", message, toolResults });
				if (options.maxSteps !== undefined && stepsUsed >= options.maxSteps && hasMoreToolCalls) {
					await emit({ type: "agent_end", messages: newMessages });
					return {
						status: "limited",
						reason: "maxSteps",
						toolResults: getToolResultMessages(producedToolOutcomes),
					};
				}
				stepRepeatAttempts = 0;
				pendingMessages = (await config.getSteeringMessages?.()) || [];
				continue;
			} else {
				await emit({ type: "message_end", message });
			}

			await emit({ type: "turn_end", message, toolResults: [] });
			stepRepeatAttempts = 0;

			pendingMessages = (await config.getSteeringMessages?.()) || [];
		}

		// Agent would stop here. Check for follow-up messages.
		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			// Set as pending so inner loop processes them
			pendingMessages = followUpMessages;
			continue;
		}

		// No more messages, exit
		break;
	}

	await emit({ type: "agent_end", messages: newMessages });
	return {
		status: getDispatchStatus("success", producedToolOutcomes),
		toolResults: getToolResultMessages(producedToolOutcomes),
	};
}

async function runBranch(
	request: DispatchRequest,
	currentContext: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	runtime: LoopRuntime,
): Promise<DispatchResult> {
	const requestedDepth = request.maxDepth ?? runtime.remainingDispatchDepth;
	const allowedDepth = Math.min(requestedDepth, runtime.remainingDispatchDepth);
	if (allowedDepth <= 0) {
		return {
			status: "limited",
			reason: "maxDepth",
			messages: [],
			toolResults: [],
		};
	}

	const baseContext = request.from === "root" ? runtime.rootContext : currentContext;
	const branchRootContext = shallowCloneAgentContext(baseContext);
	const branchContext = shallowCloneAgentContext(baseContext);
	branchContext.messages = [...baseContext.messages, ...request.messages];
	const branchMessages: AgentMessage[] = [...request.messages];
	const branchConfig: AgentLoopConfig = {
		...config,
		getSteeringMessages: undefined,
		getFollowUpMessages: undefined,
	};

	try {
		const completion = await runLoop(
			branchContext,
			branchMessages,
			branchConfig,
			signal,
			async () => {},
			runtime.streamFn,
			{
				rootContext: branchRootContext,
				remainingDispatchDepth: allowedDepth - 1,
				maxSteps: request.maxSteps,
			},
		);
		return {
			status: completion.status,
			reason: completion.reason,
			messages: branchMessages,
			toolResults: completion.toolResults,
		};
	} catch (error) {
		return {
			status: "failure",
			reason: error instanceof Error ? error.message : String(error),
			messages: branchMessages,
			toolResults: branchMessages.slice(request.messages.length).filter(isToolResultMessage),
			error,
		};
	}
}

function getDispatchStatus(status: DispatchStatus, toolOutcomes: ToolResultOutcome[] = []): DispatchStatus {
	if (status === "limited" || status === "aborted") {
		return status;
	}
	if (toolOutcomes.some((toolOutcome) => toolOutcome.status === "aborted")) {
		return "aborted";
	}
	if (toolOutcomes.some((toolOutcome) => toolOutcome.status === "limited")) {
		return "limited";
	}
	if (status === "success" && toolOutcomes.some((toolOutcome) => toolOutcome.status === "failure")) {
		return "failure";
	}
	return status;
}

function getToolResultMessages(toolOutcomes: ToolResultOutcome[]): ToolResultMessage[] {
	return toolOutcomes.map((toolOutcome) => toolOutcome.message);
}

function shallowCloneAgentContext(context: AgentContext): AgentContext {
	return {
		...context,
		messages: [...context.messages],
		tools: context.tools ? [...context.tools] : undefined,
	};
}

function isToolResultMessage(message: AgentMessage): message is ToolResultMessage {
	return message.role === "toolResult";
}

function getAssistantErrorMessage(message: AssistantMessage): string | undefined {
	return (message as AssistantMessage & { errorMessage?: string }).errorMessage;
}

/**
 * Stream an assistant response from the LLM.
 * This is where AgentMessage[] gets transformed to Message[] for the LLM.
 */
async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	streamFn?: StreamFn,
): Promise<AssistantMessage> {
	// Apply context transform if configured (AgentMessage[] → AgentMessage[])
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	// Convert to LLM-compatible messages (AgentMessage[] → Message[])
	const llmMessages = await config.convertToLlm(messages);

	// Build LLM context
	const llmContext: Context = {
		systemPrompt: context.systemPrompt,
		messages: llmMessages,
		tools: context.tools,
	};

	const streamFunction = streamFn || streamSimple;

	// Resolve API key (important for expiring tokens)
	const resolvedApiKey =
		(config.getApiKey ? await config.getApiKey(config.model.provider) : undefined) || config.apiKey;

	const response = await streamFunction(config.model, llmContext, {
		...config,
		apiKey: resolvedApiKey,
		signal,
	});

	let partialMessage: AssistantMessage | null = null;
	let addedPartial = false;

	for await (const event of response) {
		switch (event.type) {
			case "start":
				partialMessage = event.partial;
				context.messages.push(partialMessage);
				addedPartial = true;
				await emit({ type: "message_start", message: { ...partialMessage } });
				break;

			case "text_start":
			case "text_delta":
			case "text_end":
			case "thinking_start":
			case "thinking_delta":
			case "thinking_end":
			case "toolcall_start":
			case "toolcall_delta":
			case "toolcall_end":
				if (partialMessage) {
					partialMessage = event.partial;
					context.messages[context.messages.length - 1] = partialMessage;
					await emit({
						type: "message_update",
						assistantMessageEvent: event,
						message: { ...partialMessage },
					});
				}
				break;

			case "done":
			case "error": {
				const finalMessage = await response.result();
				if (addedPartial) {
					context.messages[context.messages.length - 1] = finalMessage;
				} else {
					context.messages.push(finalMessage);
				}
				if (!addedPartial) {
					await emit({ type: "message_start", message: { ...finalMessage } });
				}
				return finalMessage;
			}
		}
	}

	const finalMessage = await response.result();
	if (addedPartial) {
		context.messages[context.messages.length - 1] = finalMessage;
	} else {
		context.messages.push(finalMessage);
		await emit({ type: "message_start", message: { ...finalMessage } });
	}
	return finalMessage;
}

/**
 * Execute tool calls from an assistant message.
 */
async function executeToolCalls(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	stepRepeatAttempts: number,
	runtime: LoopRuntime,
): Promise<ExecutedToolCallBatch> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");
	const hasSequentialToolCall = toolCalls.some(
		(tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
	);
	if (config.toolExecution === "sequential" || hasSequentialToolCall) {
		return executeToolCallsSequential(
			currentContext,
			assistantMessage,
			toolCalls,
			config,
			signal,
			emit,
			stepRepeatAttempts,
			runtime,
		);
	}
	return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit, stepRepeatAttempts, runtime);
}

type ExecutedToolCallBatch = {
	outcomes: ToolResultOutcome[];
	terminate: boolean;
	retry: "none" | "step";
	injectMessages: AgentMessage[];
};

type ToolResultOutcome = {
	message: ToolResultMessage;
	status: DispatchStatus;
};

type FinalizedToolCallDecision =
	| {
			retry: "none";
			finalized: FinalizedToolCallOutcome;
	  }
	| {
			retry: "step";
			injectMessages: AgentMessage[];
	  };

type ToolCallAttemptAction = {
	retry: "none" | "tool" | "step";
	injectMessages: AgentMessage[];
};

async function executeToolCallsSequential(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	stepRepeatAttempts: number,
	runtime: LoopRuntime,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallOutcome[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal, runtime);
		let finalized: FinalizedToolCallOutcome;
		if (preparation.kind === "immediate") {
			finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
				status: preparation.status,
			};
		} else {
			const decision = await executePreparedToolCallWithRetries(
				currentContext,
				assistantMessage,
				preparation,
				config,
				signal,
				emit,
				stepRepeatAttempts,
			);
			if (decision.retry === "step") {
				return {
					outcomes: [],
					terminate: false,
					retry: "step",
					injectMessages: decision.injectMessages,
				};
			}
			finalized = decision.finalized;
		}

		await emitToolExecutionEnd(finalized, emit);
		finalizedCalls.push(finalized);
	}

	return {
		outcomes: finalizedCalls.map(createToolResultOutcome),
		terminate: shouldTerminateToolBatch(finalizedCalls),
		retry: "none",
		injectMessages: [],
	};
}

async function executeToolCallsParallel(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCalls: AgentToolCall[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	stepRepeatAttempts: number,
	runtime: LoopRuntime,
): Promise<ExecutedToolCallBatch> {
	const finalizedCalls: FinalizedToolCallEntry[] = [];
	const finalizedInCompletionOrder: FinalizedToolCallOutcome[] = [];

	for (const toolCall of toolCalls) {
		await emit({
			type: "tool_execution_start",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			args: toolCall.arguments,
		});

		const preparation = await prepareToolCall(currentContext, assistantMessage, toolCall, config, signal, runtime);
		if (preparation.kind === "immediate") {
			const finalized = {
				toolCall,
				result: preparation.result,
				isError: preparation.isError,
				status: preparation.status,
			} satisfies FinalizedToolCallOutcome;
			finalizedInCompletionOrder.push(finalized);
			finalizedCalls.push({ retry: "none", finalized });
			continue;
		}

		finalizedCalls.push(async () => {
			const decision = await executePreparedToolCallWithRetries(
				currentContext,
				assistantMessage,
				preparation,
				config,
				signal,
				emit,
				stepRepeatAttempts,
			);
			if (decision.retry === "step") {
				return decision;
			}
			finalizedInCompletionOrder.push(decision.finalized);
			return { retry: "none", finalized: decision.finalized };
		});
	}

	const orderedDecisions = await Promise.all(
		finalizedCalls.map((entry) => (typeof entry === "function" ? entry() : Promise.resolve(entry))),
	);
	const stepRetry = orderedDecisions.find((decision) => decision.retry === "step");
	if (stepRetry) {
		return {
			outcomes: [],
			terminate: false,
			retry: "step",
			injectMessages: stepRetry.injectMessages,
		};
	}

	const orderedFinalizedCalls = orderedDecisions.flatMap((decision) =>
		decision.retry === "none" ? [decision.finalized] : [],
	);
	for (const finalized of finalizedInCompletionOrder) {
		await emitToolExecutionEnd(finalized, emit);
	}

	return {
		outcomes: orderedFinalizedCalls.map(createToolResultOutcome),
		terminate: shouldTerminateToolBatch(orderedFinalizedCalls),
		retry: "none",
		injectMessages: [],
	};
}

type PreparedToolCall = {
	kind: "prepared";
	toolCall: AgentToolCall;
	tool: AgentTool<any>;
	args: unknown;
};

type ImmediateToolCallOutcome = {
	kind: "immediate";
	result: AgentToolResult<any>;
	isError: boolean;
	status?: DispatchStatus;
};

type ExecutedToolCallOutcome = {
	result: AgentToolResult<any>;
	isError: boolean;
};

type FinalizedToolCallOutcome = {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
	status?: DispatchStatus;
};

type FinalizedToolCallEntry = FinalizedToolCallDecision | (() => Promise<FinalizedToolCallDecision>);

function shouldTerminateToolBatch(finalizedCalls: FinalizedToolCallOutcome[]): boolean {
	return finalizedCalls.length > 0 && finalizedCalls.every((finalized) => finalized.result.terminate === true);
}

function prepareToolCallArguments(tool: AgentTool<any>, toolCall: AgentToolCall): AgentToolCall {
	if (!tool.prepareArguments) {
		return toolCall;
	}
	const preparedArguments = tool.prepareArguments(toolCall.arguments);
	if (preparedArguments === toolCall.arguments) {
		return toolCall;
	}
	return {
		...toolCall,
		arguments: preparedArguments as Record<string, any>,
	};
}

async function prepareToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	toolCall: AgentToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	runtime: LoopRuntime,
): Promise<PreparedToolCall | ImmediateToolCallOutcome> {
	const tool = currentContext.tools?.find((t) => t.name === toolCall.name);
	if (!tool) {
		return {
			kind: "immediate",
			result: createErrorToolResult(`Tool ${toolCall.name} not found`),
			isError: true,
			status: "failure",
		};
	}

	try {
		const preparedToolCall = prepareToolCallArguments(tool, toolCall);
		const validatedArgs = validateToolArguments(tool, preparedToolCall);
		const hookContext = {
			assistantMessage,
			toolCall,
			args: validatedArgs,
			context: currentContext,
		};

		if (config.beforeToolCall) {
			const beforeResult = await config.beforeToolCall(hookContext, signal);
			if (beforeResult?.block) {
				return {
					kind: "immediate",
					result: createErrorToolResult(beforeResult.reason || "Tool execution was blocked"),
					isError: true,
					status: "failure",
				};
			}
		}

		if (config.handleToolCall) {
			const handlerResult = await config.handleToolCall(
				{
					...hookContext,
					dispatch: (request) => runBranch(request, currentContext, config, signal, runtime),
				},
				signal,
			);
			if (!handlerResult) {
				return {
					kind: "prepared",
					toolCall,
					tool,
					args: validatedArgs,
				};
			}
			if (handlerResult.status !== "execute") {
				return {
					kind: "immediate",
					result: handlerResult.result,
					isError: handlerResult.status !== "success",
					status: handlerResult.status,
				};
			}
			if (handlerResult.arguments !== undefined) {
				const nextToolCall = prepareToolCallArguments(tool, {
					...preparedToolCall,
					arguments: handlerResult.arguments as Record<string, any>,
				});
				return {
					kind: "prepared",
					toolCall: nextToolCall,
					tool,
					args: validateToolArguments(tool, nextToolCall),
				};
			}
		}

		return {
			kind: "prepared",
			toolCall,
			tool,
			args: validatedArgs,
		};
	} catch (error) {
		return {
			kind: "immediate",
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
			status: "failure",
		};
	}
}

async function executePreparedToolCall(
	prepared: PreparedToolCall,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
): Promise<ExecutedToolCallOutcome> {
	const updateEvents: Promise<void>[] = [];

	try {
		const result = await prepared.tool.execute(
			prepared.toolCall.id,
			prepared.args as never,
			signal,
			(partialResult) => {
				updateEvents.push(
					Promise.resolve(
						emit({
							type: "tool_execution_update",
							toolCallId: prepared.toolCall.id,
							toolName: prepared.toolCall.name,
							args: prepared.toolCall.arguments,
							partialResult,
						}),
					),
				);
			},
		);
		await Promise.all(updateEvents);
		return { result, isError: false };
	} catch (error) {
		await Promise.all(updateEvents);
		return {
			result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
			isError: true,
		};
	}
}

async function executePreparedToolCallWithRetries(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	emit: AgentEventSink,
	stepRepeatAttempts: number,
): Promise<FinalizedToolCallDecision> {
	let attempt = 1;

	while (true) {
		const executed = await executePreparedToolCall(prepared, signal, emit);
		let action: ToolCallAttemptAction;
		try {
			action = await getToolCallAttemptAction(currentContext, assistantMessage, prepared, executed, attempt, config, signal);
		} catch (error) {
			const hookFailed = {
				result: createErrorToolResult(error instanceof Error ? error.message : String(error)),
				isError: true,
			};
			const finalized = await finalizeExecutedToolCall(
				currentContext,
				assistantMessage,
				prepared,
				hookFailed,
				config,
				signal,
			);
			return { retry: "none", finalized };
		}

		if (action.retry === "step") {
			if (stepRepeatAttempts >= MAX_STEP_REPEAT_ATTEMPTS) {
				const retryLimitExceeded = {
					result: createErrorToolResult(
						`Assistant step retry limit exceeded after ${stepRepeatAttempts} retries`,
					),
					isError: true,
				};
				const finalized = await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					prepared,
					retryLimitExceeded,
					config,
					signal,
				);
				return { retry: "none", finalized };
			}
			return {
				retry: "step",
				injectMessages: action.injectMessages,
			};
		}

		if (action.retry === "tool") {
			if (attempt >= MAX_TOOL_EXECUTION_ATTEMPTS) {
				const retryLimitExceeded = {
					result: createErrorToolResult(`Tool execution retry limit exceeded after ${attempt} attempts`),
					isError: true,
				};
				const finalized = await finalizeExecutedToolCall(
					currentContext,
					assistantMessage,
					prepared,
					retryLimitExceeded,
					config,
					signal,
				);
				return { retry: "none", finalized };
			}
			attempt++;
			continue;
		}

		const finalized = await finalizeExecutedToolCall(
			currentContext,
			assistantMessage,
			prepared,
			executed,
			config,
			signal,
		);
		return { retry: "none", finalized };
	}
}

async function getToolCallAttemptAction(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	attempt: number,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<ToolCallAttemptAction> {
	if (!config.afterToolExecution) {
		return { retry: "none", injectMessages: [] };
	}

	const result = await config.afterToolExecution(
		{
			assistantMessage,
			toolCall: prepared.toolCall,
			args: prepared.args,
			attempt,
			result: executed.result,
			isError: executed.isError,
			context: currentContext,
		},
		signal,
	);
	return {
		retry: result?.retry ?? "none",
		injectMessages: result?.injectMessages ?? [],
	};
}

async function finalizeExecutedToolCall(
	currentContext: AgentContext,
	assistantMessage: AssistantMessage,
	prepared: PreparedToolCall,
	executed: ExecutedToolCallOutcome,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
): Promise<FinalizedToolCallOutcome> {
	let result = executed.result;
	let isError = executed.isError;

	if (config.afterToolCall) {
		try {
			const afterResult = await config.afterToolCall(
				{
					assistantMessage,
					toolCall: prepared.toolCall,
					args: prepared.args,
					result,
					isError,
					context: currentContext,
				},
				signal,
			);
			if (afterResult) {
				result = {
					content: afterResult.content ?? result.content,
					details: afterResult.details ?? result.details,
					terminate: afterResult.terminate ?? result.terminate,
				};
				isError = afterResult.isError ?? isError;
			}
		} catch (error) {
			result = createErrorToolResult(error instanceof Error ? error.message : String(error));
			isError = true;
		}
	}

	return {
		toolCall: prepared.toolCall,
		result,
		isError,
		status: isError ? "failure" : "success",
	};
}

function createErrorToolResult(message: string): AgentToolResult<any> {
	return {
		content: [{ type: "text", text: message }],
		details: {},
	};
}

async function emitToolExecutionEnd(finalized: FinalizedToolCallOutcome, emit: AgentEventSink): Promise<void> {
	await emit({
		type: "tool_execution_end",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		result: finalized.result,
		isError: finalized.isError,
	});
}

function createToolResultMessage(finalized: FinalizedToolCallOutcome): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: finalized.toolCall.id,
		toolName: finalized.toolCall.name,
		content: finalized.result.content,
		details: finalized.result.details,
		isError: finalized.isError,
		timestamp: Date.now(),
	};
}

function createToolResultOutcome(finalized: FinalizedToolCallOutcome): ToolResultOutcome {
	return {
		message: createToolResultMessage(finalized),
		status: finalized.status ?? (finalized.isError ? "failure" : "success"),
	};
}

async function emitToolResultMessage(toolResultMessage: ToolResultMessage, emit: AgentEventSink): Promise<void> {
	await emit({ type: "message_start", message: toolResultMessage });
	await emit({ type: "message_end", message: toolResultMessage });
}
