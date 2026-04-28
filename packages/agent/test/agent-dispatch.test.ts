import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
	type ToolResultMessage,
	type UserMessage,
} from "@mariozechner/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { agentLoop } from "../src/agent-loop.js";
import type { AgentContext, AgentLoopConfig, AgentMessage, AgentTool, AgentToolResult } from "../src/types.js";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

const taskSchema = Type.Object({ task: Type.String() });
const valueSchema = Type.Object({ value: Type.String() });

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function textResult(text: string, details: unknown = {}): AgentToolResult<any> {
	return {
		content: [{ type: "text", text }],
		details,
	};
}

function textAssistant(text: string, stopReason: AssistantMessage["stopReason"] = "stop"): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function toolAssistant(id: string, name: string, args: Record<string, unknown>): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "toolCall", id, name, arguments: args }],
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason: "toolUse",
		timestamp: Date.now(),
	};
}

function user(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

function toolResultMessage(id: string, name: string, text: string, isError: boolean): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: id,
		toolName: name,
		content: [{ type: "text", text }],
		details: {},
		isError,
		timestamp: Date.now(),
	};
}

function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
}

function toolResultText(message: ToolResultMessage | undefined): string {
	return (
		message?.content
			.filter((part): part is { type: "text"; text: string } => part.type === "text")
			.map((part) => part.text)
			.join("\n") ?? ""
	);
}

function createTool(
	name: string,
	parameters: AgentTool<any>["parameters"],
	execute: (args: any) => AgentToolResult<any> | Promise<AgentToolResult<any>>,
): AgentTool<any, any> {
	return {
		name,
		label: name,
		description: name,
		parameters,
		async execute(_toolCallId, args) {
			return execute(args);
		},
	};
}

function createMacroTool(execute: (args: { task: string }) => AgentToolResult<any> | Promise<AgentToolResult<any>> = (args) =>
	textResult(`fallback:${args.task}`),
): AgentTool<any, any> {
	return createTool("macro_tool", taskSchema, execute);
}

async function runScenario(options: {
	tools: AgentTool<any>[];
	config?: Partial<AgentLoopConfig>;
	responses: AssistantMessage[];
	prompt?: UserMessage;
}) {
	const llmContexts: Message[][] = [];
	const context: AgentContext = {
		systemPrompt: "",
		messages: [],
		tools: options.tools,
	};
	const config: AgentLoopConfig = {
		model: createModel(),
		convertToLlm: (messages) => {
			const llmMessages = identityConverter(messages);
			llmContexts.push(llmMessages);
			return llmMessages;
		},
		...options.config,
	};
	let llmCalls = 0;
	const stream = agentLoop([options.prompt ?? user("parent request")], context, config, undefined, () => {
		const message = options.responses[llmCalls] ?? textAssistant("parent done");
		llmCalls++;
		const mockStream = new MockAssistantStream();
		queueMicrotask(() => {
			mockStream.push({
				type: "done",
				reason: message.stopReason === "toolUse" ? "toolUse" : "stop",
				message,
			});
		});
		return mockStream;
	});

	for await (const _event of stream) {
		// consume
	}

	return {
		llmCalls,
		llmContexts,
		messages: await stream.result(),
	};
}

function findToolResult(messages: AgentMessage[], id = "macro-call"): ToolResultMessage | undefined {
	return messages.find(
		(message): message is ToolResultMessage => message.role === "toolResult" && message.toolCallId === id,
	);
}

describe("agent dispatch branches", () => {
	it("dispatches from root with isolated branch messages", async () => {
		const subtask = user("Sub-task: solve the isolated micro problem.");
		const { llmCalls, llmContexts, messages } = await runScenario({
			tools: [createMacroTool()],
			config: {
				handleToolCall: async (hook) => {
					const result = await hook.dispatch({ from: "root", messages: [subtask], maxSteps: 1, maxDepth: 1 });
					return { status: "success", result: textResult(`assembled:${result.status}`, { result }) };
				},
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "macro" }),
				textAssistant("branch root result"),
				textAssistant("parent done"),
			],
		});

		expect(llmCalls).toBe(3);
		expect(llmContexts[1]).toEqual([subtask]);
		expect(toolResultText(findToolResult(messages))).toBe("assembled:success");
	});

	it("dispatches from current with explicit appended messages", async () => {
		const pending = toolResultMessage("macro-call", "macro_tool", "Pending recursive results.", false);
		const subtask = user("Sub-task: solve only this split part.");
		const { llmCalls, llmContexts, messages } = await runScenario({
			tools: [createMacroTool()],
			config: {
				handleToolCall: async (hook) => {
					const result = await hook.dispatch({
						from: "current",
						messages: [pending, subtask],
						maxSteps: 1,
						maxDepth: 1,
					});
					return { status: "success", result: textResult(`assembled:${result.status}`, { result }) };
				},
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "macro" }),
				textAssistant("branch current result"),
				textAssistant("parent done"),
			],
		});

		const macroAssistant = messages.find(
			(message) =>
				message.role === "assistant" &&
				message.content.some((part) => part.type === "toolCall" && part.id === "macro-call"),
		);
		expect(llmCalls).toBe(3);
		expect(llmContexts[1]).toEqual([messages[0], macroAssistant, pending, subtask]);
		expect(toolResultText(findToolResult(messages))).toBe("assembled:success");
	});

	it("derives status from branch-produced tool results, not seed transcript tool results", async () => {
		const seedError = toolResultMessage("prior-error", "prior_tool", "prior failure kept for context", true);
		const { messages } = await runScenario({
			tools: [createMacroTool()],
			config: {
				handleToolCall: async (hook) => {
					const result = await hook.dispatch({
						from: "current",
						messages: [seedError, user("Sub-task: finish successfully despite prior context.")],
						maxSteps: 1,
						maxDepth: 1,
					});
					const visibleToolResults = result.messages.filter((message) => message.role === "toolResult").length;
					return {
						status: "success",
						result: textResult(`assembled:${result.status}:${result.toolResults.length}:${visibleToolResults}`, {
							result,
						}),
					};
				},
			},
			responses: [toolAssistant("macro-call", "macro_tool", { task: "macro" }), textAssistant("branch succeeded")],
		});

		expect(toolResultText(findToolResult(messages))).toBe("assembled:success:0:1");
	});

	it("returns limited for maxSteps and maxDepth", async () => {
		const executed: string[] = [];
		const stepTool = createTool("step_tool", valueSchema, (args) => {
			executed.push(args.value);
			return textResult(`step:${args.value}`, args);
		});
		const maxSteps = await runScenario({
			tools: [createMacroTool(), stepTool],
			config: {
				handleToolCall: async (hook) => {
					if (hook.toolCall.name !== "macro_tool") return undefined;
					const result = await hook.dispatch({
						from: "root",
						messages: [user("Sub-task: keep working.")],
						maxSteps: 1,
						maxDepth: 1,
					});
					return {
						status: "success",
						result: textResult(
							`steps:${result.status}:${result.reason}:${result.toolResults.length}:${result.toolResults[0]?.isError}`,
							{ result },
						),
					};
				},
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "macro" }),
				toolAssistant("branch-tool", "step_tool", { value: "final-step" }),
			],
		});
		const maxDepth = await runScenario({
			tools: [createMacroTool()],
			config: {
				handleToolCall: async (hook) => {
					const result = await hook.dispatch({
						from: "root",
						messages: [user("Sub-task: should not start.")],
						maxDepth: 0,
					});
					return { status: "success", result: textResult(`depth:${result.status}:${result.reason}`, { result }) };
				},
			},
			responses: [toolAssistant("macro-call", "macro_tool", { task: "macro" }), textAssistant("parent done")],
		});

		expect(executed).toEqual(["final-step"]);
		expect(toolResultText(findToolResult(maxSteps.messages))).toBe("steps:limited:maxSteps:1:false");
		expect(toolResultText(findToolResult(maxDepth.messages))).toBe("depth:limited:maxDepth");
		expect(maxDepth.llmCalls).toBe(2);
	});

	it("does not let beforeToolCall replace tool execution", async () => {
		const executed: string[] = [];
		const macroTool = createMacroTool((args) => {
			executed.push(args.task);
			return textResult(`real:${args.task}`, args);
		});
		const { messages } = await runScenario({
			tools: [macroTool],
			config: {
				beforeToolCall: async () =>
					({
						result: textResult("ignored"),
					}) as any,
			},
			responses: [toolAssistant("macro-call", "macro_tool", { task: "original" }), textAssistant("parent done")],
		});

		expect(executed).toEqual(["original"]);
		expect(toolResultText(findToolResult(messages))).toBe("real:original");
	});

	it("propagates handled nested statuses without returning isError", async () => {
		const { messages } = await runScenario({
			tools: [createMacroTool()],
			config: {
				handleToolCall: async (hook) => {
					const nested = await hook.dispatch({
						from: "root",
						messages: [user("Sub-task: should be limited before running.")],
						maxDepth: 0,
					});
					return {
						status: nested.status,
						result: textResult(`assembled:${nested.status}`, { nested }),
					};
				},
			},
			responses: [toolAssistant("macro-call", "macro_tool", { task: "macro" }), textAssistant("parent done")],
		});

		const macroResult = findToolResult(messages);
		expect(toolResultText(macroResult)).toBe("assembled:limited");
		expect(macroResult?.isError).toBe(true);
	});

	it("reports failure for finalized branch tool errors", async () => {
		const failingTool = createTool("failing_tool", valueSchema, (args) => {
			throw new Error(`boom:${args.value}`);
		});
		const { messages } = await runScenario({
			tools: [createMacroTool(), failingTool],
			config: {
				handleToolCall: async (hook) => {
					if (hook.toolCall.name !== "macro_tool") return undefined;
					const result = await hook.dispatch({
						from: "root",
						messages: [user("Sub-task: call the failing tool.")],
						maxSteps: 3,
						maxDepth: 1,
					});
					return {
						status: "success",
						result: textResult(`assembled:${result.status}:${result.toolResults.length}:${result.toolResults[0]?.isError}`, {
							result,
						}),
					};
				},
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "macro" }),
				toolAssistant("branch-fail", "failing_tool", { value: "bad" }),
				textAssistant("branch observed failure"),
			],
		});

		expect(toolResultText(findToolResult(messages))).toBe("assembled:failure:1:true");
	});

	it("reports success when retries or afterToolCall repair a branch tool failure", async () => {
		let attempts = 0;
		const flakyTool = createTool("flaky_tool", valueSchema, (args) => {
			attempts++;
			if (attempts === 1) throw new Error(`transient:${args.value}`);
			return textResult(`ok:${args.value}`, args);
		});
		const repairedTool = createTool("repaired_tool", valueSchema, (args) => {
			throw new Error(`repairable:${args.value}`);
		});
		const commonConfig = {
			handleToolCall: async (hook: any) => {
				if (hook.toolCall.name !== "macro_tool") return undefined;
				const result = await hook.dispatch({
					from: "root",
					messages: [user(`Sub-task: call ${hook.args.task}.`)],
					maxSteps: 3,
					maxDepth: 1,
				});
				return {
					status: "success",
					result: textResult(`assembled:${result.status}:${result.toolResults.length}:${result.toolResults[0]?.isError}`, {
						result,
					}),
				};
			},
		};

		const retried = await runScenario({
			tools: [createMacroTool(), flakyTool],
			config: {
				...commonConfig,
				afterToolExecution: async (hook) =>
					hook.toolCall.name === "flaky_tool" && hook.isError ? { retry: "tool" } : { retry: "none" },
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "flaky_tool" }),
				toolAssistant("branch-flaky", "flaky_tool", { value: "eventual" }),
				textAssistant("branch recovered"),
			],
		});
		const repaired = await runScenario({
			tools: [createMacroTool(), repairedTool],
			config: {
				...commonConfig,
				afterToolCall: async (hook) =>
					hook.toolCall.name === "repaired_tool" && hook.isError
						? { content: [{ type: "text", text: "repaired" }], details: { repaired: true }, isError: false }
						: undefined,
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "repaired_tool" }),
				toolAssistant("branch-repaired", "repaired_tool", { value: "fix" }),
				textAssistant("branch repaired"),
			],
		});

		expect(attempts).toBe(2);
		expect(toolResultText(findToolResult(retried.messages))).toBe("assembled:success:1:false");
		expect(toolResultText(findToolResult(repaired.messages))).toBe("assembled:success:1:false");
	});

	it("returns aborted when a branch assistant response aborts", async () => {
		const { messages } = await runScenario({
			tools: [createMacroTool()],
			config: {
				handleToolCall: async (hook) => {
					const result = await hook.dispatch({
						from: "root",
						messages: [user("Sub-task: abort.")],
						maxSteps: 1,
						maxDepth: 1,
					});
					return {
						status: "success",
						result: textResult(`assembled:${result.status}:${result.toolResults.length}`, { result }),
					};
				},
			},
			responses: [
				toolAssistant("macro-call", "macro_tool", { task: "macro" }),
				textAssistant("branch aborted", "aborted"),
				textAssistant("parent done"),
			],
		});

		expect(toolResultText(findToolResult(messages))).toBe("assembled:aborted:0");
	});

	it("lets handleToolCall continue to the original tool with edited arguments", async () => {
		let handlerCalls = 0;
		const executed: string[] = [];
		const macroTool = createMacroTool((args) => {
			executed.push(args.task);
			return textResult(`real:${args.task}`, args);
		});
		const { messages } = await runScenario({
			tools: [macroTool],
			config: {
				handleToolCall: async (hook) => {
					handlerCalls++;
					return { status: "execute", arguments: { task: "edited" } };
				},
			},
			responses: [toolAssistant("macro-call", "macro_tool", { task: "original" }), textAssistant("parent done")],
		});

		expect(handlerCalls).toBe(1);
		expect(executed).toEqual(["edited"]);
		expect(toolResultText(findToolResult(messages))).toBe("real:edited");
	});

	it("runs afterToolCall for tools continued through execute status", async () => {
		const macroTool = createMacroTool((args) => textResult(`real:${args.task}`, args));
		const { messages } = await runScenario({
			tools: [macroTool],
			config: {
				handleToolCall: async () => ({ status: "execute", arguments: { task: "edited" } }),
				afterToolCall: async (hook) => ({
					content: [{ type: "text", text: `after:${toolResultText({ content: hook.result.content } as ToolResultMessage)}` }],
					details: hook.result.details,
				}),
			},
			responses: [toolAssistant("macro-call", "macro_tool", { task: "original" }), textAssistant("parent done")],
		});

		expect(toolResultText(findToolResult(messages))).toBe("after:real:edited");
	});
});
