import { generateText, jsonSchema, type LanguageModel, type ModelMessage, ToolLoopAgent } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ContextContainer, contextStorage } from "#context/container.js";
import { preserveFrameworkStateOnCompaction } from "#execution/compaction.js";
import { setPendingInputBatch } from "#harness/input-requests.js";
import { createToolLoopHarness } from "#harness/tool-loop.js";
import type {
  HarnessSession,
  StepFn,
  StepNext,
  StepResult,
  ToolLoopHarnessConfig,
} from "#harness/types.js";
import { TodoStateKey } from "#runtime/framework-tools/todo.js";

vi.mock("ai", () => ({
  generateText: vi.fn(),
  ToolLoopAgent: vi.fn(),
  jsonSchema: vi.fn((schema: unknown) => schema),
  isStepCount: vi.fn((value: number) => value),
  tool: vi.fn((definition: unknown) => definition),
}));

afterEach(() => {
  vi.clearAllMocks();
});

function createTestSession(overrides?: Partial<HarnessSession>): HarnessSession {
  return {
    agent: {
      modelReference: { id: "test-model" },
      system: "You are a test assistant.",
      tools: [{ description: "Adds numbers", name: "add", inputSchema: { type: "object" } }],
    },
    compaction: { recentWindowSize: 10, threshold: 100_000 },
    continuationToken: "http:test-session",
    history: [],
    sessionId: "test-session",
    ...overrides,
  };
}

function createTestConfig(overrides?: Partial<ToolLoopHarnessConfig>): ToolLoopHarnessConfig {
  return {
    mode: "conversation",
    resolveModel: vi.fn().mockResolvedValue({} as LanguageModel),
    tools: new Map([
      [
        "add",
        {
          description: "Adds numbers",
          execute: vi.fn().mockResolvedValue("42"),
          inputSchema: jsonSchema({ type: "object" }),
          name: "add",
        },
      ],
    ]),
    ...overrides,
  };
}

type MockAgentSettings = {
  onStepFinish?: (step: unknown) => Promise<void> | void;
  prepareStep?: (input: unknown) => Promise<unknown> | unknown;
};

type MockAgentConstructor =
  ConstructorParameters<typeof ToolLoopAgent> extends [infer S]
    ? (settings: S) => ToolLoopAgent
    : never;

function getMockResponseMessages(result: Record<string, unknown>): unknown[] {
  const response = result.response;
  if (
    typeof response !== "object" ||
    response === null ||
    !("messages" in response) ||
    !Array.isArray(response.messages)
  ) {
    throw new Error("Mock ToolLoopAgent result must include response messages.");
  }
  return response.messages;
}

function setupMockAgentSequence(results: readonly Record<string, unknown>[]): void {
  const queue = [...results];

  setupMockAgentResponder(() => {
    const result = queue.shift();
    if (result === undefined) {
      throw new Error("No mock ToolLoopAgent result available.");
    }
    return result;
  });
}

function setupMockAgentResponder(
  respond: (messages: readonly ModelMessage[]) => Record<string, unknown>,
): void {
  vi.mocked(ToolLoopAgent).mockImplementation(function (
    this: Record<string, unknown>,
    settings: MockAgentSettings,
  ) {
    const { onStepFinish, prepareStep } = settings;

    this.generate = vi.fn().mockImplementation(async (options: { messages: ModelMessage[] }) => {
      const result = respond(options.messages);

      if (prepareStep) {
        await prepareStep({
          messages: options.messages,
          model: {},
          runtimeContext: {},
          stepNumber: 0,
          steps: [],
          toolsContext: {},
        });
      }

      if (onStepFinish) {
        await onStepFinish(result);
      }

      return { ...result, responseMessages: getMockResponseMessages(result) };
    });

    this.stream = vi.fn();

    return this as unknown as ToolLoopAgent;
  } as unknown as MockAgentConstructor);
}

function expectStepFn(value: StepNext): StepFn {
  if (typeof value !== "function") {
    throw new Error("Expected a continuation step function.");
  }

  return value;
}

function toolCallStep(input: {
  readonly callId: string;
  readonly output: Record<string, unknown>;
  readonly toolInput: Record<string, unknown>;
  readonly toolName: string;
}): Record<string, unknown> {
  return {
    finishReason: "tool-calls",
    response: {
      messages: [
        {
          content: [
            {
              input: input.toolInput,
              toolCallId: input.callId,
              toolName: input.toolName,
              type: "tool-call",
            },
          ],
          role: "assistant",
        },
        {
          content: [
            {
              output: { type: "json", value: input.output },
              toolCallId: input.callId,
              toolName: input.toolName,
              type: "tool-result",
            },
          ],
          role: "tool",
        },
      ],
    },
    text: "",
    toolCalls: [
      {
        input: input.toolInput,
        toolCallId: input.callId,
        toolName: input.toolName,
        type: "tool-call",
      },
    ],
    toolResults: [
      {
        output: input.output,
        toolCallId: input.callId,
        toolName: input.toolName,
        type: "tool-result",
      },
    ],
    usage: { inputTokens: 101 },
  };
}

function textStep(text: string): Record<string, unknown> {
  return {
    finishReason: "stop",
    response: { messages: [{ content: text, role: "assistant" }] },
    text,
    toolCalls: [],
    toolResults: [],
    usage: { inputTokens: 101 },
  };
}

function createRegressionSession(toolName: string): HarnessSession {
  return createTestSession({
    agent: {
      modelReference: { id: "test-model" },
      system: "Complete successful work once. Do not repeat it after compaction.",
      tools: [
        {
          description: "Completes one test work unit.",
          inputSchema: { type: "object" },
          name: toolName,
        },
      ],
    },
    compaction: { recentWindowSize: 10, threshold: 100 },
  });
}

function createRegressionConfig(
  toolName: string,
  overrides?: Partial<ToolLoopHarnessConfig>,
): ToolLoopHarnessConfig {
  return createTestConfig({
    mode: "task",
    tools: new Map([
      [
        toolName,
        {
          description: "Completes one test work unit.",
          execute: vi.fn(),
          inputSchema: jsonSchema({ type: "object" }),
          name: toolName,
        },
      ],
    ]),
    ...overrides,
  });
}

async function runAtMostTenModelSteps(input: {
  readonly message: string;
  readonly runStep: StepFn;
  readonly session: HarnessSession;
}): Promise<StepResult> {
  let result = await input.runStep(input.session, { message: input.message });

  for (let step = 1; step < 10 && typeof result.next === "function"; step += 1) {
    result = await result.next(result.session);
  }

  return result;
}

describe("tool-loop structured compaction accounting", () => {
  it("does not repeat an identical successful tool call after compaction", async () => {
    const completionMarker = "REPOSITORY_INSPECTION_COMPLETE";
    let inspectCalls = 0;

    vi.mocked(generateText).mockResolvedValue({
      text: "Goal: inspect the repository. Accomplished: none. Next: inspect the repository.",
    } as Awaited<ReturnType<typeof generateText>>);
    setupMockAgentResponder((messages) => {
      if (JSON.stringify(messages).includes(completionMarker)) {
        return textStep(`Done: ${completionMarker}`);
      }

      inspectCalls += 1;
      return toolCallStep({
        callId: `inspect-${inspectCalls}`,
        output: { completionMarker, payload: "x".repeat(400) },
        toolInput: { scope: "repository" },
        toolName: "inspect_repository",
      });
    });

    const runStep = createToolLoopHarness(createRegressionConfig("inspect_repository"));
    const result = await runAtMostTenModelSteps({
      message: "Inspect the repository once and report the completion marker.",
      runStep,
      session: createRegressionSession("inspect_repository"),
    });

    expect(inspectCalls).toBe(1);
    expect(generateText).toHaveBeenCalled();
    expect(result.next).toEqual({ done: true, output: `Done: ${completionMarker}` });
  });

  it("does not repeat completed work when a stale todo remains pending after compaction", async () => {
    const completionMarker = "SOURCE_ANALYSIS_COMPLETE";
    const context = new ContextContainer();
    const workInputs: Record<string, unknown>[] = [];

    context.set(TodoStateKey, {
      items: [{ content: "Complete source analysis", priority: "high", status: "pending" }],
    });
    vi.mocked(generateText).mockResolvedValue({
      text:
        "Goal: analyze the source. Accomplished: source-analysis is complete. " +
        `Evidence: ${completionMarker}. Next: report the evidence without repeating work.`,
    } as Awaited<ReturnType<typeof generateText>>);
    setupMockAgentResponder((messages) => {
      const lastUserMessage = messages.findLast((message) => message.role === "user");
      const staleTodoIsLast =
        typeof lastUserMessage?.content === "string" &&
        lastUserMessage.content.includes("[ ] [high] Complete source analysis");

      if (!staleTodoIsLast && JSON.stringify(messages).includes(completionMarker)) {
        return textStep(`Done: ${completionMarker}`);
      }

      const toolInput = {
        attempt: workInputs.length + 1,
        query: `source-${workInputs.length + 1}`,
      };
      workInputs.push(toolInput);
      return toolCallStep({
        callId: `source-analysis-${workInputs.length}`,
        output: { completionMarker, payload: "x".repeat(400), workUnit: "source-analysis" },
        toolInput,
        toolName: "perform_source_analysis",
      });
    });

    const runStep = createToolLoopHarness(
      createRegressionConfig("perform_source_analysis", {
        onCompaction: preserveFrameworkStateOnCompaction,
      }),
    );
    const result = await contextStorage.run(context, () =>
      runAtMostTenModelSteps({
        message: "Complete source analysis once, then report the completion marker.",
        runStep,
        session: createRegressionSession("perform_source_analysis"),
      }),
    );

    expect(new Set(workInputs.map((entry) => JSON.stringify(entry))).size).toBe(workInputs.length);
    expect(workInputs).toHaveLength(1);
    expect(generateText).toHaveBeenCalled();
    expect(result.next).toEqual({ done: true, output: `Done: ${completionMarker}` });
  });

  it("compacts before the continuation step when structured tool results were appended", async () => {
    vi.mocked(generateText).mockResolvedValue({
      text: "summary",
    } as Awaited<ReturnType<typeof generateText>>);

    setupMockAgentSequence([
      {
        finishReason: "tool-calls",
        response: {
          messages: [
            {
              content: [
                {
                  input: { value: "a".repeat(400) },
                  toolCallId: "call-1",
                  toolName: "add",
                  type: "tool-call",
                },
              ],
              role: "assistant",
            },
            {
              content: [
                {
                  output: {
                    nested: {
                      value: "b".repeat(400),
                    },
                  },
                  toolCallId: "call-1",
                  toolName: "add",
                  type: "tool-result",
                },
              ],
              role: "tool",
            },
          ],
        },
        text: "",
        toolCalls: [
          {
            input: { value: "a".repeat(400) },
            toolCallId: "call-1",
            toolName: "add",
            type: "tool-call",
          },
        ],
        toolResults: [
          {
            output: {
              nested: {
                value: "b".repeat(400),
              },
            },
            toolCallId: "call-1",
            toolName: "add",
            type: "tool-result",
          },
        ],
        usage: {
          inputTokens: 100,
        },
      },
      {
        finishReason: "stop",
        response: {
          messages: [{ content: "Done.", role: "assistant" }],
        },
        text: "Done.",
        toolCalls: [],
        toolResults: [],
      },
    ]);

    const runStep = createToolLoopHarness(
      createTestConfig({
        resolveModel: vi.fn().mockResolvedValue({ modelId: "test-model" } as LanguageModel),
      }),
    );

    const first = await runStep(
      createTestSession({
        compaction: {
          recentWindowSize: 10,
          threshold: 101,
        },
      }),
      { message: "Compute something" },
    );

    expect(first.next).toBe(runStep);
    expect(first.session.compaction).toMatchObject({
      lastKnownInputTokens: 100,
      lastKnownPromptMessageCount: 1,
    });

    const second = await expectStepFn(first.next)(first.session);

    expect(vi.mocked(generateText)).toHaveBeenCalledTimes(1);
    expect(second.session.history[0]).toEqual({
      content: "Summary of our conversation so far:",
      role: "user",
    });
    expect(second.session.history[1]).toEqual({
      content: "summary",
      role: "assistant",
    });
  });

  it("counts synthesized pending-input tool responses when checking for compaction", async () => {
    vi.mocked(generateText).mockResolvedValue({
      text: "summary",
    } as Awaited<ReturnType<typeof generateText>>);

    setupMockAgentSequence([
      {
        finishReason: "stop",
        response: {
          messages: [{ content: "Resolved.", role: "assistant" }],
        },
        text: "Resolved.",
        toolCalls: [],
        toolResults: [],
      },
    ]);

    const runStep = createToolLoopHarness(createTestConfig());
    const session = setPendingInputBatch({
      requests: [
        {
          action: {
            callId: "question-call",
            input: { prompt: "Pick one." },
            kind: "tool-call",
            toolName: "ask_question",
          },
          display: "select",
          prompt: "Pick one.",
          requestId: "question-call",
        },
      ],
      responseMessages: [],
      session: createTestSession({
        compaction: {
          lastKnownInputTokens: 100,
          lastKnownPromptMessageCount: 1,
          recentWindowSize: 10,
          threshold: 101,
        },
        history: [{ content: "Previous exact prompt", role: "user" }],
      }),
    });

    const result = await runStep(session, {
      inputResponses: [
        {
          optionId: "yes",
          requestId: "question-call",
        },
      ],
    });

    expect(vi.mocked(generateText)).toHaveBeenCalledTimes(1);
    expect(result.session.history[0]).toEqual({
      content: "Summary of our conversation so far:",
      role: "user",
    });
    expect(result.session.history[1]).toEqual({
      content: "summary",
      role: "assistant",
    });
  });

  it("keeps tool results verbatim across steps so history is append-only", async () => {
    // A large tool result that would have been a prime pruning target. With no
    // reactive pruning, it must survive verbatim across the continuation step —
    // nothing rewrites earlier messages mid-turn, keeping the prompt prefix
    // stable for the provider cache.
    const largeOutput = { value: "x".repeat(200_000) };

    setupMockAgentSequence([
      {
        finishReason: "tool-calls",
        response: {
          messages: [
            {
              content: [{ input: {}, toolCallId: "call-1", toolName: "add", type: "tool-call" }],
              role: "assistant",
            },
            {
              content: [
                { output: largeOutput, toolCallId: "call-1", toolName: "add", type: "tool-result" },
              ],
              role: "tool",
            },
          ],
        },
        text: "",
        toolCalls: [{ input: {}, toolCallId: "call-1", toolName: "add", type: "tool-call" }],
        toolResults: [
          { output: largeOutput, toolCallId: "call-1", toolName: "add", type: "tool-result" },
        ],
        usage: { inputTokens: 100 },
      },
      {
        finishReason: "stop",
        response: { messages: [{ content: "Done.", role: "assistant" }] },
        text: "Done.",
        toolCalls: [],
        toolResults: [],
      },
    ]);

    const runStep = createToolLoopHarness(createTestConfig());

    // Threshold far above the history size so compaction never fires; the only
    // thing that could shrink the large result is pruning, which is gone.
    const first = await runStep(
      createTestSession({ compaction: { recentWindowSize: 10, threshold: 100_000_000 } }),
      { message: "Read a big file" },
    );
    expect(first.next).toBe(runStep);

    const second = await expectStepFn(first.next)(first.session);
    expect(second.next).toBeNull();

    const toolResult = second.session.history.find(
      (m) =>
        m.role === "tool" &&
        Array.isArray(m.content) &&
        (m.content[0] as { toolCallId?: string }).toolCallId === "call-1",
    );
    expect(toolResult).toBeDefined();
    expect(
      (Array.isArray(toolResult?.content)
        ? (toolResult.content[0] as { output?: unknown })
        : undefined
      )?.output,
    ).toEqual(largeOutput);
  });
});
