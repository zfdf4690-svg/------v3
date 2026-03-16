import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// 将数据库文件放在项目根目录
const dbPath = path.resolve(__dirname, '../data.db'); 

export const db = new Database(dbPath);

db.pragma('foreign_keys = ON'); // 启用外键支持

// 1. 创建 meetings 表
db.exec(`
  CREATE TABLE IF NOT EXISTS meetings (
    id TEXT PRIMARY KEY,
    title TEXT,
    startTime TEXT,
    status TEXT
  )
`);

// 2. 创建 knowledgeBase 表
db.exec(`
  CREATE TABLE IF NOT EXISTS knowledgeBase (
    id TEXT PRIMARY KEY,
    meetingId TEXT,
    title TEXT,
    date TEXT,
    summary TEXT,
    keyTakeaways TEXT, 
    actionItems TEXT,  
    participants TEXT, 
    tags TEXT,         
    FOREIGN KEY (meetingId) REFERENCES meetings(id) ON DELETE CASCADE
  )
`);

// 3. 创建 actionItems 表
db.exec(`
  DROP TABLE IF EXISTS actionItems;
  CREATE TABLE actionItems (
    id TEXT PRIMARY KEY,
    meetingId TEXT,
    task TEXT,
    status TEXT,
    assignee TEXT,
    dueDate TEXT,
    FOREIGN KEY (meetingId) REFERENCES meetings(id) ON DELETE CASCADE
  )
`);

// 加基本索引 (meetingId)
db.exec(`
  CREATE INDEX IF NOT EXISTS idx_knowledgeBase_meetingId ON knowledgeBase(meetingId);
  CREATE INDEX IF NOT EXISTS idx_actionItems_meetingId ON actionItems(meetingId);
`);
