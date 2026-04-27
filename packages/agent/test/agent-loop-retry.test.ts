import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
	type UserMessage,
} from "@mariozechner/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { agentLoop } from "../src/agent-loop.js";
import type { AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool } from "../src/types.js";

// Mock stream for testing - mimics MockAssistantStream
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

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createUserMessage(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

// Simple identity converter for tests - just passes through standard messages
function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
}

function branchLabel(message: AgentMessage | Message): string | undefined {
	if (message.role === "user") {
		return `user:${typeof message.content === "string" ? message.content : JSON.stringify(message.content)}`;
	}
	if (message.role === "assistant") {
		return `assistant:${message.content
			.filter((part) => part.type === "toolCall")
			.map((part) => part.id)
			.join(",")}`;
	}
	if (message.role === "toolResult") {
		return `toolResult:${message.toolCallId}`;
	}
	return undefined;
}

function branchLabels(messages: Array<AgentMessage | Message>): string[] {
	return messages.flatMap((message) => {
		const label = branchLabel(message);
		return label ? [label] : [];
	});
}

function committedMessageEndLabels(events: AgentEvent[]): string[] {
	return events.flatMap((event) => {
		if (event.type !== "message_end") {
			return [];
		}
		const label = branchLabel(event.message);
		return label ? [label] : [];
	});
}

describe("agentLoop retry behavior", () => {
	it("should let afterToolExecution retry a failed tool execution locally", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const attemptsSeen: Array<{ attempt: number; isError: boolean }> = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "flaky",
			label: "Flaky",
			description: "Fails once",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				if (executed.length === 1) {
					throw new Error("transient failure");
				}
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolExecution: async ({ attempt, isError }) => {
				attemptsSeen.push({ attempt, isError });
				return { retry: isError && attempt === 1 ? "tool" : "none" };
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("run flaky")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "flaky", arguments: { value: "hello" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolResults = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message];
		});

		expect(executed).toEqual(["hello", "hello"]);
		expect(attemptsSeen).toEqual([
			{ attempt: 1, isError: true },
			{ attempt: 2, isError: false },
		]);
		expect(toolResults).toHaveLength(1);
		expect(toolResults[0].isError).toBe(false);
		expect(toolResults[0].content).toEqual([{ type: "text", text: "ok:hello" }]);
	});

	it("should report afterToolExecution hook failures as error tool results", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolExecution: async () => {
				throw new Error("afterToolExecution failed");
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("run echo")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolResults = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message];
		});

		expect(toolResults).toHaveLength(1);
		expect(toolResults[0].isError).toBe(true);
		expect(toolResults[0].content).toEqual([{ type: "text", text: "afterToolExecution failed" }]);
	});

	it("should let afterToolExecution request an assistant step retry with injectMessages", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const llmContexts: Message[][] = [];
		const hintMessage: AgentMessage = createUserMessage("HINT: call the tool with value=repaired");
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "validator",
			label: "Validator",
			description: "Validates input",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				if (params.value === "bad") {
					throw new Error("bad value");
				}
				return {
					content: [{ type: "text", text: `accepted:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				const llmMessages = identityConverter(messages);
				llmContexts.push(llmMessages);
				return llmMessages;
			},
			afterToolExecution: async ({ isError }) => {
				if (!isError) {
					return { retry: "none" };
				}
				return {
					retry: "step",
					injectMessages: [hintMessage],
				};
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("validate something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-bad", name: "validator", arguments: { value: "bad" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else if (llmCalls === 2) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-good", name: "validator", arguments: { value: "repaired" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		const roles = messages.map((message) => message.role);
		const toolResultIds = messages.flatMap((message) =>
			message.role === "toolResult" ? [message.toolCallId] : [],
		);
		const secondLlmContext = llmContexts[1];
		const branchContexts = llmContexts.map(branchLabels);
		const committedMessageEnds = committedMessageEndLabels(events);

		expect(executed).toEqual(["bad", "repaired"]);
		expect(llmCalls).toBe(3);
		expect(secondLlmContext.map((message) => message.role)).toEqual(["user", "user"]);
		expect(secondLlmContext[1]).toEqual(hintMessage);
		expect(branchContexts).toEqual([
			["user:validate something"],
			["user:validate something", "user:HINT: call the tool with value=repaired"],
			[
				"user:validate something",
				"user:HINT: call the tool with value=repaired",
				"assistant:tool-good",
				"toolResult:tool-good",
			],
		]);
		expect(committedMessageEnds).toEqual([
			"user:validate something",
			"user:HINT: call the tool with value=repaired",
			"assistant:tool-good",
			"toolResult:tool-good",
			"assistant:",
		]);
		expect(roles).toEqual(["user", "user", "assistant", "toolResult", "assistant"]);
		expect(toolResultIds).toEqual(["tool-good"]);
	});

	it("should stop repeating assistant step retries after the retry limit", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "always_bad",
			label: "Always Bad",
			description: "Always fails",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				throw new Error("still bad");
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolExecution: async () => ({
				retry: "step",
				injectMessages: [createUserMessage("try again")],
			}),
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("keep retrying")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls <= 11) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: `tool-${llmCalls}`, name: "always_bad", arguments: { value: "bad" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolResults = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message];
		});

		expect(executed).toHaveLength(11);
		expect(llmCalls).toBe(12);
		expect(toolResults).toHaveLength(1);
		expect(toolResults[0].isError).toBe(true);
		expect(toolResults[0].content).toEqual([
			{ type: "text", text: "Assistant step retry limit exceeded after 10 retries" },
		]);
	});

	it("should discard the whole assistant step when any tool execution requests step retry", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const llmContexts: Message[][] = [];
		const hintMessage: AgentMessage = createUserMessage("HINT: regenerate the whole tool batch");
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "batch_tool",
			label: "Batch Tool",
			description: "Batch tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				if (params.value === "bad") {
					throw new Error("bad value");
				}
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				const llmMessages = identityConverter(messages);
				llmContexts.push(llmMessages);
				return llmMessages;
			},
			toolExecution: "sequential",
			afterToolExecution: async ({ isError }) => {
				if (!isError) {
					return { retry: "none" };
				}
				return {
					retry: "step",
					injectMessages: [hintMessage],
				};
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("run batch")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-ok-before-retry", name: "batch_tool", arguments: { value: "ok-before" } },
							{ type: "toolCall", id: "tool-bad", name: "batch_tool", arguments: { value: "bad" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else if (llmCalls === 2) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-ok-after-retry", name: "batch_tool", arguments: { value: "ok-after" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		const toolResultIds = messages.flatMap((message) =>
			message.role === "toolResult" ? [message.toolCallId] : [],
		);
		const branchContexts = llmContexts.map(branchLabels);
		const committedMessageEnds = committedMessageEndLabels(events);
		const executionStarts = events.flatMap((event) =>
			event.type === "tool_execution_start" ? [`start:${event.toolCallId}`] : [],
		);
		const executionEnds = events.flatMap((event) =>
			event.type === "tool_execution_end" ? [`end:${event.toolCallId}:${event.isError}`] : [],
		);

		expect(executed).toEqual(["ok-before", "bad", "ok-after"]);
		expect(llmCalls).toBe(3);
		expect(messages).toContainEqual(hintMessage);
		expect(branchContexts).toEqual([
			["user:run batch"],
			["user:run batch", "user:HINT: regenerate the whole tool batch"],
			[
				"user:run batch",
				"user:HINT: regenerate the whole tool batch",
				"assistant:tool-ok-after-retry",
				"toolResult:tool-ok-after-retry",
			],
		]);
		expect(committedMessageEnds).toEqual([
			"user:run batch",
			"user:HINT: regenerate the whole tool batch",
			"assistant:tool-ok-after-retry",
			"toolResult:tool-ok-after-retry",
			"assistant:",
		]);
		expect(toolResultIds).toEqual(["tool-ok-after-retry"]);
		expect(executionStarts).toEqual([
			"start:tool-ok-before-retry",
			"start:tool-bad",
			"start:tool-ok-after-retry",
		]);
		expect(executionEnds).toEqual(["end:tool-ok-before-retry:false", "end:tool-ok-after-retry:false"]);
	});

	it("should inject failed tool output before reissuing the assistant step", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const llmContexts: Message[][] = [];
		let injectedMessage: AgentMessage | undefined;
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "validator",
			label: "Validator",
			description: "Validates input",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				if (params.value === "bad") {
					throw new Error("validation failed: expected value=good");
				}
				return {
					content: [{ type: "text", text: `accepted:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				const llmMessages = identityConverter(messages);
				llmContexts.push(llmMessages);
				return llmMessages;
			},
			afterToolExecution: async ({ isError, result }) => {
				if (!isError) {
					return { retry: "none" };
				}
				const toolOutput = result.content
					.filter((part): part is { type: "text"; text: string } => part.type === "text")
					.map((part) => part.text)
					.join("\n");
				injectedMessage = createUserMessage(`HINT from failed tool output: ${toolOutput}`);
				return {
					retry: "step",
					injectMessages: [injectedMessage],
				};
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("validate with output hint")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-bad", name: "validator", arguments: { value: "bad" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else if (llmCalls === 2) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-good", name: "validator", arguments: { value: "good" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		const toolResultIds = messages.flatMap((message) =>
			message.role === "toolResult" ? [message.toolCallId] : [],
		);
		const branchContexts = llmContexts.map(branchLabels);
		const committedMessageEnds = committedMessageEndLabels(events);

		expect(executed).toEqual(["bad", "good"]);
		expect(llmCalls).toBe(3);
		expect(injectedMessage).toBeDefined();
		expect(llmContexts[1]).toContainEqual(injectedMessage);
		expect((injectedMessage as Extract<AgentMessage, { role: "user" }>).content).toContain(
			"validation failed: expected value=good",
		);
		expect(branchContexts).toEqual([
			["user:validate with output hint"],
			["user:validate with output hint", branchLabel(injectedMessage!)],
			[
				"user:validate with output hint",
				branchLabel(injectedMessage!),
				"assistant:tool-good",
				"toolResult:tool-good",
			],
		]);
		expect(committedMessageEnds).toEqual([
			"user:validate with output hint",
			branchLabel(injectedMessage!),
			"assistant:tool-good",
			"toolResult:tool-good",
			"assistant:",
		]);
		expect(toolResultIds).toEqual(["tool-good"]);
	});

	it("should inject two tool-name examples before reissuing the assistant step after the first failure", async () => {
		const toolSchema = Type.Object({ expression: Type.String() });
		const executed: string[] = [];
		const llmContexts: Message[][] = [];
		let injectedMessage: AgentMessage | undefined;
		const tool: AgentTool<typeof toolSchema, { expression: string }> = {
			name: "calculator",
			label: "Calculator",
			description: "Evaluates expressions",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.expression);
				if (params.expression === "bad") {
					throw new Error("invalid expression");
				}
				return {
					content: [{ type: "text", text: `result:${params.expression}` }],
					details: { expression: params.expression },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				const llmMessages = identityConverter(messages);
				llmContexts.push(llmMessages);
				return llmMessages;
			},
			afterToolExecution: async ({ isError, attempt, toolCall }) => {
				if (!isError || attempt !== 1) {
					return { retry: "none" };
				}
				injectedMessage = createUserMessage(
					[
						`EXAMPLE 1 for ${toolCall.name}: use expression="1 + 1"`,
						`EXAMPLE 2 for ${toolCall.name}: use expression="2 * 3"`,
					].join("\n"),
				);
				return {
					retry: "step",
					injectMessages: [injectedMessage],
				};
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("calculate with examples")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-bad", name: "calculator", arguments: { expression: "bad" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else if (llmCalls === 2) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-good", name: "calculator", arguments: { expression: "1 + 1" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}
		const branchContexts = llmContexts.map(branchLabels);
		const committedMessageEnds = committedMessageEndLabels(events);

		expect(executed).toEqual(["bad", "1 + 1"]);
		expect(llmCalls).toBe(3);
		expect(injectedMessage).toBeDefined();
		expect(llmContexts[1]).toContainEqual(injectedMessage);
		expect((injectedMessage as Extract<AgentMessage, { role: "user" }>).content).toContain(
			"EXAMPLE 1 for calculator",
		);
		expect((injectedMessage as Extract<AgentMessage, { role: "user" }>).content).toContain(
			"EXAMPLE 2 for calculator",
		);
		expect(branchContexts).toEqual([
			["user:calculate with examples"],
			["user:calculate with examples", branchLabel(injectedMessage!)],
			[
				"user:calculate with examples",
				branchLabel(injectedMessage!),
				"assistant:tool-good",
				"toolResult:tool-good",
			],
		]);
		expect(committedMessageEnds).toEqual([
			"user:calculate with examples",
			branchLabel(injectedMessage!),
			"assistant:tool-good",
			"toolResult:tool-good",
			"assistant:",
		]);
	});

	it("should locally retry a tool three times and commit only the final success", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const attemptsSeen: Array<{ attempt: number; isError: boolean }> = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "flaky_k3",
			label: "Flaky K3",
			description: "Fails twice",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				if (executed.length < 3) {
					throw new Error(`transient failure ${executed.length}`);
				}
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			systemPrompt: "",
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolExecution: async ({ attempt, isError }) => {
				attemptsSeen.push({ attempt, isError });
				return { retry: isError && attempt < 3 ? "tool" : "none" };
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("run flaky k3")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-k3", name: "flaky_k3", arguments: { value: "hello" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolResults = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message];
		});

		expect(executed).toEqual(["hello", "hello", "hello"]);
		expect(attemptsSeen).toEqual([
			{ attempt: 1, isError: true },
			{ attempt: 2, isError: true },
			{ attempt: 3, isError: false },
		]);
		expect(llmCalls).toBe(2);
		expect(toolResults).toHaveLength(1);
		expect(toolResults[0].isError).toBe(false);
		expect(toolResults[0].content).toEqual([{ type: "text", text: "ok:hello" }]);
	});
});
