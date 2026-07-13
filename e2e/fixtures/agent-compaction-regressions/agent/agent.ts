import { defineAgent, defineDynamic, type DynamicResolveContext } from "eve";
import { mockModel } from "eve/evals";

const TEST_CONTEXT_WINDOW_TOKENS = 32_000;

const compactionModel = mockModel({
  modelId: "compaction-regression-checkpoint",
  respond({ lastUserMessage, toolResults }) {
    const reachedAttemptCap = toolResults.some(
      ({ output }) =>
        typeof output === "object" &&
        output !== null &&
        "hardStop" in output &&
        output.hardStop === true,
    );

    if (lastUserMessage?.includes("[case: stale-todo-work]")) {
      return [
        "[case: stale-todo-work]",
        "Goal: complete source analysis once.",
        "Accomplished: workUnit source-analysis is complete.",
        "Evidence: SOURCE_ANALYSIS_COMPLETE.",
        reachedAttemptCap
          ? "Hard stop: 10 attempts reached. Report the evidence and call no more tools."
          : "Next: report the evidence. Do not call perform-source-analysis again.",
      ].join("\n");
    }

    return [
      "[case: redundant-tool-calls]",
      "Goal: inspect the repository once.",
      reachedAttemptCap ? "Hard stop: 10 attempts reached." : "Accomplished: none.",
      reachedAttemptCap
        ? "Report REPOSITORY_INSPECTION_COMPLETE and call no more tools."
        : "Open work: call inspect-repository, then report REPOSITORY_INSPECTION_COMPLETE.",
    ].join("\n");
  },
});

export default defineAgent({
  model: defineDynamic({
    fallback: "anthropic/claude-sonnet-5",
    events: {
      "turn.started": (_event, ctx) => selectTestModel(lastUserText(ctx.messages)),
    },
  }),
  modelContextWindowTokens: TEST_CONTEXT_WINDOW_TOKENS,
  compaction: {
    model: compactionModel,
    modelContextWindowTokens: TEST_CONTEXT_WINDOW_TOKENS,
    thresholdPercent: 0.001,
  },
  limits: {
    maxInputTokensPerSession: 100_000,
  },
});

function selectTestModel(text: string) {
  if (text.includes("[model: gpt-5.6]")) {
    return { model: "openai/gpt-5.6-sol", modelContextWindowTokens: TEST_CONTEXT_WINDOW_TOKENS };
  }
  if (text.includes("[model: opus-4.8]")) {
    return {
      model: "anthropic/claude-opus-4.8",
      modelContextWindowTokens: TEST_CONTEXT_WINDOW_TOKENS,
    };
  }
  if (text.includes("[model: sonnet-5]")) {
    return {
      model: "anthropic/claude-sonnet-5",
      modelContextWindowTokens: TEST_CONTEXT_WINDOW_TOKENS,
    };
  }
  return null;
}

function lastUserText(messages: DynamicResolveContext["messages"]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role !== "user") continue;
    if (typeof message.content === "string") return message.content;
    return message.content.map((part) => (part.type === "text" ? part.text : "")).join(" ");
  }
  return "";
}
