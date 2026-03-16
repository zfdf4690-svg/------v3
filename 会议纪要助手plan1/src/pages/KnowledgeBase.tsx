import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Search, 
  Calendar, 
  Users, 
  Tag, 
  ChevronDown, 
  MessageSquare, 
  Lightbulb, 
  Send, 
  Paperclip, 
  Mic,
  Bot
} from 'lucide-react';
import { useMeetings } from '../contexts/MeetingContext';

export interface KnowledgeBaseEntry {
  id: string;
  meetingId: string;
  title: string;
  date: string;
  summary: string;
  keyTakeaways: string[];
  actionItems: { task: string; status: string; assignee: string; dueDate: string }[];
  participants: { id: string; name: string }[];
  tags: string[];
}

export function KnowledgeBase() {
  const { knowledgeBaseEntries } = useMeetings();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const filteredEntries = (knowledgeBaseEntries || []).filter(entry => 
    entry.title.toLowerCase().includes(searchQuery.toLowerCase())
  );
  const [messages, setMessages] = useState<{ role: 'user' | 'ai', text: string }[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!userInput.trim()) return;

    const userMessage = { role: 'user' as const, text: userInput };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);

    try {
      // 提取核心上下文，抛弃不必要字段节约 Token
      const contextData = ((knowledgeBaseEntries as any[]) || []).map(e => ({
        title: e.title,
        summary: e.summary,
        actionItems: e.actionItems || []
      }));
      
      const prompt = `你是一个会议纪要助手。请基于以下知识库内容回答用户问题。\n\n知识库内容：\n${JSON.stringify(contextData)}\n\n用户问题: ${userInput}`;
      const res = await fetch('/api/qwen-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      if (!res.ok) {
         throw new Error('API fetch failed');
      }

      const data = await res.json();
      setMessages(prev => [...prev, { role: 'ai', text: data.text || '无法生成回答。' }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: '抱歉，处理您的请求时出错。' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-1 overflow-hidden h-[calc(100vh-4rem)]">
      {/* Left Content Area */}
      <div className="flex-1 flex flex-col overflow-y-auto px-8 py-6 max-w-5xl mx-auto w-full">
        {/* Search Section */}
        <div className="mb-8">
          <div className="relative group">
            <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none text-slate-400">
              <Search className="w-5 h-5" />
            </div>
            <input 
              type="text" 
              placeholder="搜索会议记录、决策或待办事项..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl py-4 pl-12 pr-4 text-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent transition-all shadow-sm outline-none" 
            />
          </div>
          <div className="flex items-center gap-3 mt-4 text-sm">
            <span className="text-slate-500">最近搜索:</span>
            <div className="flex gap-2">
              <span className="px-3 py-1 bg-slate-100 dark:bg-slate-800 rounded-full cursor-pointer hover:bg-blue-600/10 hover:text-blue-600 transition-colors border border-transparent hover:border-blue-600/20">季度规划</span>
              <span className="px-3 py-1 bg-slate-100 dark:bg-slate-800 rounded-full cursor-pointer hover:bg-blue-600/10 hover:text-blue-600 transition-colors border border-transparent hover:border-blue-600/20">产品迭代</span>
              <span className="px-3 py-1 bg-slate-100 dark:bg-slate-800 rounded-full cursor-pointer hover:bg-blue-600/10 hover:text-blue-600 transition-colors border border-transparent hover:border-blue-600/20">技术方案评审</span>
            </div>
          </div>
        </div>

        {/* Filter Section */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
              <Calendar className="w-4 h-4" />
              按日期
              <ChevronDown className="w-4 h-4" />
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
              <Users className="w-4 h-4" />
              按参会人
              <ChevronDown className="w-4 h-4" />
            </button>
            <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
              <Tag className="w-4 h-4" />
              按关键词
              <ChevronDown className="w-4 h-4" />
            </button>
          </div>
          <div className="text-sm text-slate-500">
            共找到 <span className="text-blue-600 font-semibold">{124 + (knowledgeBaseEntries || []).length}</span> 个结果
          </div>
        </div>

        {/* Results Area */}
        <div className="space-y-4 pb-12">
          {filteredEntries
            .map((e: any) => ({
              ...e,
              date: e.date || new Date().toISOString().split('T')[0],
              summary: e.summary || '暂无摘要',
              participants: e.participants || [{ id: '1', name: '系统' }],
              actionItems: e.actionItems || [],
              tags: e.tags || ['会议记录']
            }))
            .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
            .map((entry: KnowledgeBaseEntry) => (
            <div key={entry.id} onClick={() => navigate(`/meetings/${entry.meetingId}`)} className="p-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl hover:shadow-md hover:border-blue-600/30 transition-all cursor-pointer group flex flex-col gap-3">
              <div className="flex justify-between items-start">
                <h3 className="text-lg font-bold group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">{entry.title}</h3>
                <span className="text-xs font-medium text-slate-500 px-2 py-1 bg-slate-100 dark:bg-slate-800 rounded">{entry.date}</span>
              </div>
              
              <p className="text-sm text-slate-600 dark:text-slate-400 line-clamp-2">
                {entry.summary}
              </p>

              <div className="flex items-center justify-between mt-2 pt-3 border-t border-slate-100 dark:border-slate-800">
                <div className="flex items-center gap-4">
                  <div className="flex -space-x-2">
                    {entry.participants.slice(0, 3).map((p, i) => (
                      <div key={i} className="w-6 h-6 rounded-full border-2 border-white dark:border-slate-900 bg-blue-600 flex items-center justify-center text-[10px] text-white" title={p.name}>
                        {p.name.charAt(0)}
                      </div>
                    ))}
                    {entry.participants.length > 3 && (
                      <div className="w-6 h-6 rounded-full border-2 border-white dark:border-slate-900 bg-slate-200 dark:bg-slate-800 flex items-center justify-center text-[10px] text-slate-600">
                        +{entry.participants.length - 3}
                      </div>
                    )}
                  </div>
                  <span className="text-xs text-slate-500 flex items-center gap-1">
                     <Calendar className="w-3 h-3" /> {entry.actionItems.length} 个行动项
                  </span>
                </div>
                
                <div className="flex gap-1.5 flex-wrap justify-end">
                  {entry.tags.map(tag => (
                    <span key={tag} className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 bg-blue-600/10 text-blue-600 dark:text-blue-400 rounded-full">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Right Sidebar: Ask AI */}
      <aside className="w-96 border-l border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 flex flex-col shrink-0">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-2 mb-1">
            <MessageSquare className="text-blue-600 w-5 h-5" />
            <h2 className="font-bold text-lg">询问 AI</h2>
          </div>
          <p className="text-sm text-slate-500">智能助手随时为您解答搜索结果中的细节</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-60">
              <div className="p-4 bg-blue-600/5 rounded-full">
                <Lightbulb className="text-blue-600 w-8 h-8" />
              </div>
              <div>
                <p className="text-sm font-medium mb-1">您可以试着问我：</p>
                <p className="text-xs text-slate-500 leading-relaxed italic">
                  "上周关于产品规划的会议决策是什么？"<br />
                  "总结一下关于数据库迁移的所有待办事项"<br />
                  "参会人张三在昨天的评审会中提出了哪些建议？"
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'ai' && <div className="p-2 bg-slate-100 dark:bg-slate-800 rounded-full"><Bot className="w-4 h-4 text-blue-600" /></div>}
                <div className={`p-3 rounded-lg text-sm ${msg.role === 'user' ? 'bg-blue-600 text-white' : 'bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200'}`}>
                  {msg.text}
                </div>
              </div>
            ))
          )}
          {isLoading && <div className="text-sm text-slate-500 italic">AI 正在思考...</div>}
        </div>

        {/* Chat Input Box */}
        <div className="p-6 bg-slate-50 dark:bg-slate-900/50 border-t border-slate-200 dark:border-slate-800">
          <div className="relative">
            <textarea 
              className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-3 text-sm focus:ring-2 focus:ring-blue-600 focus:border-transparent transition-all resize-none shadow-sm outline-none" 
              placeholder="询问关于这些会议的具体问题..." 
              rows={3}
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
            ></textarea>
            <button 
              onClick={handleSendMessage}
              disabled={isLoading}
              className="absolute bottom-3 right-3 bg-blue-600 text-white p-1.5 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="mt-3 flex items-center justify-between">
            <div className="flex gap-2">
              <button className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded text-slate-500">
                <Paperclip className="w-5 h-5" />
              </button>
              <button className="p-1 hover:bg-slate-200 dark:hover:bg-slate-800 rounded text-slate-500">
                <Mic className="w-5 h-5" />
              </button>
            </div>
            <span className="text-[10px] text-slate-400">GPT-4 支持生成</span>
          </div>
        </div>
      </aside>
    </div>
  );
}
