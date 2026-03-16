import 'dotenv/config';
import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// OpenAI client (DashScope) — initialised lazily so the server starts even without a key
// ---------------------------------------------------------------------------
function getGenAI(): OpenAI {
  const apiKey = process.env.DASHSCOPE_API_KEY;
  if (!apiKey) {
    throw new Error("DASHSCOPE_API_KEY environment variable is not set");
  }
  return new OpenAI({
    apiKey,
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  });
}

// ---------------------------------------------------------------------------
// Build the prompt sent to Gemini
// ---------------------------------------------------------------------------
type ChatMessage = {
  sender?: string;
  role?: string;
  text?: string;
  time?: string;
  [key: string]: unknown;
};

function buildPrompt(messages: ChatMessage[]): string {
  const transcript = messages
    .filter((m) => m.text)
    .map((m) => `[${m.time ?? ""}] ${m.sender ?? ""} (${m.role ?? ""}): ${m.text}`)
    .join("\n");

  return `你是一位专业的会议助手。请根据以下会议聊天记录，提取关键决策和行动项。

会议记录：
${transcript}

请严格按照以下 JSON 格式返回，不要包含任何额外的解释或 markdown 代码块标记：
{
  "decisions": ["决策1", "决策2"],
  "actions": ["行动项1", "行动项2"]
}

要求：
- decisions：会议中达成的决定或共识（字符串数组，每项简洁明确）
- actions：需要跟进执行的具体事项，包含负责人和截止时间（如有）
- 如没有找到对应内容，返回空数组 []`;
}

// ---------------------------------------------------------------------------
// Parse JSON from model output, stripping optional ```json … ``` fences
// ---------------------------------------------------------------------------
function parseModelJson(raw: string): { decisions: string[]; actions: string[] } {
  const cleaned = raw
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  const parsed = JSON.parse(cleaned);
  return {
    decisions: Array.isArray(parsed.decisions) ? parsed.decisions : [],
    actions: Array.isArray(parsed.actions) ? parsed.actions : [],
  };
}

function buildSummaryPrompt(messages: ChatMessage[], title: string = "未知会议"): string {
  const transcript = messages
    .filter((m) => m.text)
    .map((m) => `[${m.time ?? ""}] ${m.sender ?? ""} (${m.role ?? ""}): ${m.text}`)
    .join("\n");

  return `你是一位专业的会议助手。请根据以下会议【${title}】的聊天记录，提取关键决策和行动项。

会议记录：
${transcript}

请严格按照以下 JSON 格式返回，不要包含任何额外的解释或 markdown 代码块标记：
{
  "decisions": ["决策1", "决策2"],
  "actions": [
    {
      "task": "具体的任务内容",
      "status": "未开始",
      "assignee": "负责人名字，没提到则为 待分配",
      "dueDate": "截止日期，格式如 2023-11-01，没提到则为空"
    }
  ]
}

要求：
- decisions：会议中达成的决定或共识（字符串数组）
- actions：需要跟进执行的具体事项（对象数组）
- 如没有找到对应内容，返回空数组 []`;
}

function parseEnhancedSummaryJson(raw: string): any {
  const cleaned = raw.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  const parsed = JSON.parse(cleaned);
  return {
    decisions: Array.isArray(parsed.decisions) ? parsed.decisions : [],
    actions: Array.isArray(parsed.actions) ? parsed.actions : [],
  };
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // -------------------------------------------------------------------------
  // POST /api/sync  (unchanged mock)
  // -------------------------------------------------------------------------
  app.post("/api/sync", (_req, res) => {
    res.json({
      todos: [
        { id: '1', text: '下周三前确定供应商名单', checked: false },
        { id: '2', text: '拟定 KPI 考核体系方案', checked: false },
      ],
      activeVote: null,
    });
  });

  // -------------------------------------------------------------------------
  // POST /api/gemini-summary  — real Gemini 1.5 Flash call
  // -------------------------------------------------------------------------
  app.post("/api/gemini-summary", async (req, res) => {
    // 1. Validate request body
    const { messages } = req.body as { messages?: ChatMessage[] };
    if (!Array.isArray(messages) || messages.length === 0) {
      res.status(400).json({ error: "messages 字段必须是非空数组" });
      return;
    }

    // 2. Check API key early
    if (!process.env.DASHSCOPE_API_KEY) {
      console.error("[gemini-summary] DASHSCOPE_API_KEY is not set");
      res.status(503).json({ error: "API Key 未配置，请联系管理员" });
      return;
    }

    try {
      const ai = getGenAI();
      const prompt = buildPrompt(messages);

      console.log(`[gemini-summary] Calling Qwen with ${messages.length} messages…`);

      const response = await ai.chat.completions.create({
        model: "qwen-plus",
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
      });

      const rawText = response.choices[0]?.message?.content ?? "";
      console.log("[gemini-summary] Raw response:", rawText.slice(0, 200));

      const result = parseModelJson(rawText);
      res.json(result);
    } catch (err: unknown) {
      const error = err as { status?: number; message?: string; toString?: () => string };
      const statusCode = error?.status ?? 0;
      const message = error?.message ?? String(err);

      // Rate limit
      if (statusCode === 429 || message.includes("429") || message.toLowerCase().includes("quota")) {
        console.warn("[gemini-summary] Rate limit hit:", message);
        res.status(429).json({
          error: "请求过于频繁，请稍后再试 (Rate limit exceeded)",
          retryAfter: 60,
        });
        return;
      }

      // JSON parse failure from model output
      if (err instanceof SyntaxError) {
        console.error("[gemini-summary] JSON parse error:", err.message);
        res.status(500).json({ error: "AI 返回格式解析失败，请重试" });
        return;
      }

      console.error("[gemini-summary] Unexpected error:", err);
      res.status(500).json({ error: `总结生成失败：${message}` });
    }
  });

  // -------------------------------------------------------------------------
  // POST /api/generate-summary  — Enhanced summary generator for End Meeting
  // -------------------------------------------------------------------------
  app.post("/api/generate-summary", async (req, res) => {
    const { chatMessages, meetingId, title } = req.body;
    if (!Array.isArray(chatMessages) || chatMessages.length === 0) {
      res.status(400).json({ error: "messages不能为空" });
      return;
    }

    if (!process.env.DASHSCOPE_API_KEY) {
      res.status(503).json({ error: "API Key 未配置" });
      return;
    }

    try {
      const ai = getGenAI();
      const prompt = buildSummaryPrompt(chatMessages, title || '未命名会议');

      const response = await ai.chat.completions.create({
        model: "qwen-plus",
        messages: [{ role: "user", content: prompt }],
        response_format: { type: "json_object" },
      });

      const rawText = response.choices[0]?.message?.content ?? "";
      const result = parseEnhancedSummaryJson(rawText);
      res.json(result);
    } catch (err: unknown) {
      console.error("[generate-summary] Error:", err);
      res.status(500).json({ error: "总结生成失败" });
    }
  });

  // -------------------------------------------------------------------------
  // POST /api/qwen-chat  — real Qwen call for Knowledge Base AI Chat
  // -------------------------------------------------------------------------
  app.post("/api/qwen-chat", async (req, res) => {
    const { prompt } = req.body as { prompt?: string };
    if (!prompt) {
      res.status(400).json({ error: "prompt 字段必须提供" });
      return;
    }

    if (!process.env.DASHSCOPE_API_KEY) {
      res.status(503).json({ error: "API Key 未配置" });
      return;
    }

    try {
      const ai = getGenAI();
      const response = await ai.chat.completions.create({
        model: "qwen-plus",
        messages: [{ role: "user", content: prompt }]
      });

      const rawText = response.choices[0]?.message?.content ?? "无法生成回答。";
      res.json({ text: rawText });
    } catch (err: unknown) {
      const error = err as { message?: string };
      console.error("[qwen-chat] Error:", err);
      res.status(500).json({ error: `AI 回答失败：${error?.message || err}` });
    }
  });

  // -------------------------------------------------------------------------
  // Phase 1 API: Knowledge Base
  // -------------------------------------------------------------------------
app.get('/api/knowledge', async (_req, res) => {
    try {
      const { db } = await import('./server/db.ts');
      const entries = db.prepare('SELECT * FROM knowledgeBase').all();
      
      const formattedEntries = entries.map((e: any) => ({
        ...e,
        keyTakeaways: JSON.parse(e.keyTakeaways || '[]'),
        actionItems: JSON.parse(e.actionItems || '[]'),
        participants: JSON.parse(e.participants || '[]'),
        tags: JSON.parse(e.tags || '[]')
      }));
      res.json(formattedEntries);
    } catch (err: any) {
      console.error('[GET /api/knowledge] Error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  app.post('/api/knowledge', async (req, res) => {
    const { meetingId, title, date, summary, keyTakeaways, actionItems, participants, tags } = req.body;

    if (!meetingId || !title) {
      res.status(400).json({ error: 'meetingId and title are required' });
      return; 
    }

    try {
      const { db } = await import('./server/db.ts');
      const id = Date.now().toString(); 
      const stmt = db.prepare(`
        INSERT INTO knowledgeBase (id, meetingId, title, date, summary, keyTakeaways, actionItems, participants, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
      `);
      
      stmt.run(
        id, 
        meetingId, 
        title, 
        date || new Date().toISOString().split('T')[0], 
        summary || '',
        JSON.stringify(keyTakeaways || []),
        JSON.stringify(actionItems || []),
        JSON.stringify(participants || []),
        JSON.stringify(tags || [])
      );
      
      res.status(201).json({ success: true, id });
    } catch (err: any) {
      console.error('[POST /api/knowledge] Error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  // -------------------------------------------------------------------------
  // Phase 1 API: Action Items (Todos)
  // -------------------------------------------------------------------------
  app.get('/api/todos/:meetingId', async (req, res) => {
    try {
      const { db } = await import('./server/db.ts');
      const stmt = db.prepare('SELECT * FROM actionItems WHERE meetingId = ?');
      const items = stmt.all(req.params.meetingId);
      // SQLite returns strings, so we map it appropriately to frontend structures.
      const mapped = items.map((i: any) => ({
        id: i.id,
        text: i.task,
        checked: i.status === '已完成',
        status: i.status,
        assignee: i.assignee,
        dueDate: i.dueDate
      }));
      res.json(mapped);
    } catch (err: any) {
      console.error('[GET /api/todos] Error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  app.post('/api/todos', async (req, res) => {
    const { meetingId, task, status, assignee, dueDate } = req.body;

    if (!meetingId || !task) {
      res.status(400).json({ error: 'meetingId and task are required' });
      return;
    }

    try {
      const { db } = await import('./server/db.ts');
      const id = Date.now().toString();
      const stmt = db.prepare(`
        INSERT INTO actionItems (id, meetingId, task, status, assignee, dueDate)
        VALUES (?, ?, ?, ?, ?, ?)
      `);
      stmt.run(id, meetingId, task, status || '未开始', assignee || '待分配', dueDate || '');
      
      res.status(201).json({ success: true, id });
    } catch (err: any) {
      console.error('[POST /api/todos] Error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  // -------------------------------------------------------------------------
  // Vite middleware / static serving
  // -------------------------------------------------------------------------
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, "dist")));
    app.get("*", (_req, res) => {
      res.sendFile(path.resolve(__dirname, "dist", "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
