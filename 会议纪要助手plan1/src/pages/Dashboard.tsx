import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMeetings } from '../contexts/MeetingContext';
import { 
  Plus, 
  TrendingUp, 
  BrainCircuit, 
  Timer, 
  Users, 
  Video, 
  Clock, 
  MoreVertical, 
  ArrowRight,
  CheckCircle2,
  Play
} from 'lucide-react';

export function Dashboard() {
  const navigate = useNavigate();
  const { meetings, addMeeting } = useMeetings();
  
  const pendingMeetings = (meetings || []).filter(m => m.status === 'pending' || m.status === 'not_started').slice(0, 2);

  return (
    <div className="p-8 max-w-7xl mx-auto w-full">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h2 className="text-2xl font-bold">欢迎回来，张经理</h2>
          <p className="text-slate-500 mt-1">今天有 3 个会议等待您的参与。</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={() => addMeeting({
              id: Date.now().toString(),
              title: '实时模拟会议 - ' + new Date().toLocaleTimeString(),
              startTime: '18:30 - 19:30',
              status: 'pending',
              isHost: true,
              participants: 3,
              participantNames: ['张经理', '李研发', '王设计'],
              summary: '会议正在进行中...',
              decisions: [],
              takeaways: [],
              actionItems: [],
              votes: []
            })}
            className="bg-emerald-600 hover:bg-emerald-700 text-white px-6 py-3 rounded-xl font-bold flex items-center gap-2 shadow-lg shadow-emerald-600/25 transition-all active:scale-95"
          >
            <Plus className="w-5 h-5" />
            <span>模拟实时会议</span>
          </button>
          <button 
            onClick={() => navigate('/meetings/new')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-xl font-bold flex items-center gap-2 shadow-lg shadow-blue-600/25 transition-all active:scale-95"
          >
            <Plus className="w-5 h-5" />
            <span>开始新会议</span>
          </button>
        </div>
      </div>

      {/* Insights */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-10">
        <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 className="w-5 h-5 text-blue-600" />
            <h3 className="font-bold">待办事项</h3>
          </div>
          <div className="space-y-3">
            <label className="flex items-center gap-3 p-2 hover:bg-white/50 dark:hover:bg-slate-800/50 rounded-lg cursor-pointer transition-colors">
              <input type="checkbox" className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-600/20" />
              <span className="text-sm">整理产品策略会议的行动项 (明天)</span>
            </label>
            <label className="flex items-center gap-3 p-2 hover:bg-white/50 dark:hover:bg-slate-800/50 rounded-lg cursor-pointer transition-colors">
              <input type="checkbox" className="w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-600/20" />
              <span className="text-sm">将 TechFlow 反馈同步至 CRM 系统 (本周五)</span>
            </label>
          </div>
        </div>
        <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <h3 className="font-bold">AI 会议洞察</h3>
          </div>
          <ul className="space-y-4">
            <li className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-600 mt-2 shrink-0"></div>
              <p className="text-sm text-slate-600 dark:text-slate-400">本周会议时长较上周减少了 <span className="text-blue-600 font-semibold">15%</span>，但决策效率提升。</p>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-2 h-2 rounded-full bg-blue-600 mt-2 shrink-0"></div>
              <p className="text-sm text-slate-600 dark:text-slate-400">“成本优化”是最近被提及频率最高的主题，共计出现 <span className="text-blue-600 font-semibold">42 次</span>。</p>
            </li>
          </ul>
        </div>
      </div>

      {/* Pending Meetings */}
      <div className="mb-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <span className="w-2 h-6 bg-blue-600 rounded-full"></span>
            <h3 className="text-xl font-bold">待办会议</h3>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {pendingMeetings.map((meeting, index) => (
            <div key={meeting.id} className={`bg-white/70 dark:bg-slate-900/70 backdrop-blur-md p-6 rounded-2xl border-l-4 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-sm hover:shadow-md transition-all ${index === 0 ? 'border-l-blue-600' : 'border-l-slate-300'}`}>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded ${index === 0 ? 'text-blue-600 bg-blue-600/10' : 'text-slate-500 bg-slate-100 dark:bg-slate-800'}`}>
                    {meeting.status === 'pending' ? '即将开始' : '未开始'}
                  </span>
                  <span className="text-xs text-slate-400 font-medium">{meeting.startTime}</span>
                </div>
                <h4 className="text-lg font-bold mb-3">{meeting.title}</h4>
                <div className="flex -space-x-2">
                  <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${meeting.id}1`} className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                  <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${meeting.id}2`} className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                  {meeting.participants > 2 && (
                    <div className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-[10px] font-bold text-slate-500">+{meeting.participants - 2}</div>
                  )}
                </div>
              </div>
              <div className="flex flex-col gap-2 min-w-[140px]">
                {meeting.isHost ? (
                  <button 
                    onClick={() => navigate(`/meetings/${meeting.id}/active`)}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl font-bold text-sm shadow-md shadow-blue-600/20 transition-all flex items-center justify-center gap-2"
                  >
                    <Play className="w-5 h-5 fill-current" />
                    开播
                  </button>
                ) : (
                  <button 
                    onClick={() => navigate(`/meetings/${meeting.id}/active`)}
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl font-bold text-sm shadow-md shadow-blue-600/20 transition-all flex items-center justify-center gap-2"
                  >
                    <Video className="w-5 h-5" />
                    进入会议
                  </button>
                )}
                {index === 0 && (
                  <button className="w-full bg-blue-600/5 hover:bg-blue-600/10 text-blue-600 px-4 py-2 rounded-xl font-bold text-sm transition-all flex items-center justify-center gap-2 border border-blue-600/20">
                    <Clock className="w-5 h-5" />
                    会前提醒
                  </button>
                )}
              </div>
            </div>
          ))}
          
          {pendingMeetings.length === 0 && (
            <div className="col-span-2 text-center py-10 text-slate-500">
              暂无待办会议
            </div>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 text-blue-600 rounded-xl flex items-center justify-center">
              <Video className="w-6 h-6" />
            </div>
            <span className="text-emerald-500 text-sm font-semibold flex items-center">
              +12% <TrendingUp className="w-3 h-3 ml-1" />
            </span>
          </div>
          <p className="text-slate-500 text-sm font-medium">总会议数</p>
          <p className="text-3xl font-bold mt-1">128</p>
        </div>

        <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-indigo-100 dark:bg-indigo-900/30 text-indigo-600 rounded-xl flex items-center justify-center">
              <BrainCircuit className="w-6 h-6" />
            </div>
            <span className="text-emerald-500 text-sm font-semibold flex items-center">
              +0.5% <TrendingUp className="w-3 h-3 ml-1" />
            </span>
          </div>
          <p className="text-slate-500 text-sm font-medium">AI 准确率</p>
          <p className="text-3xl font-bold mt-1">99.2%</p>
        </div>

        <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
          <div className="flex items-center justify-between mb-4">
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900/30 text-purple-600 rounded-xl flex items-center justify-center">
              <Timer className="w-6 h-6" />
            </div>
            <span className="text-emerald-500 text-sm font-semibold flex items-center">
              +8% <TrendingUp className="w-3 h-3 ml-1" />
            </span>
          </div>
          <p className="text-slate-500 text-sm font-medium">节省时间</p>
          <p className="text-3xl font-bold mt-1">
            45 <span className="text-lg font-medium text-slate-400">小时</span>
          </p>
        </div>
      </div>

      {/* My Groups */}
      <div className="mb-10">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <span className="w-2 h-6 bg-blue-600 rounded-full"></span>
            <h3 className="text-xl font-bold">我的小组</h3>
          </div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="border-2 border-dashed border-blue-500/30 rounded-2xl p-6 flex flex-col items-center justify-center text-center hover:bg-blue-500/5 transition-colors cursor-pointer group">
            <div className="w-14 h-14 rounded-full bg-blue-500/10 text-blue-600 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Users className="w-8 h-8" />
            </div>
            <h4 className="text-lg font-bold mb-1">创建新小组</h4>
            <p className="text-sm text-slate-500 mb-4">按项目或部门组织成员，共享会议摘要和知识库。</p>
            <button className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-bold shadow-md shadow-blue-600/20">
              立即创建
            </button>
          </div>

          <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl shadow-sm hover:shadow-md transition-all">
            <div className="flex justify-between items-start mb-4">
              <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 text-blue-600 rounded-xl flex items-center justify-center">
                <Users className="w-6 h-6" />
              </div>
              <div className="flex -space-x-2">
                <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=A" className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=B" className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                <div className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100 flex items-center justify-center text-[10px] text-slate-500 font-bold">+8</div>
              </div>
            </div>
            <h4 className="text-lg font-bold mb-1">市场推广部</h4>
            <p className="text-sm text-slate-500 mb-4 truncate">主要负责 Q3 新品发布的所有会议与资料。</p>
            <div className="flex items-center justify-between text-xs font-medium">
              <span className="text-slate-400">12 个相关会议</span>
              <button className="text-blue-600 hover:underline">管理小组成员</button>
            </div>
          </div>

          <div className="bg-white/70 dark:bg-slate-900/70 backdrop-blur-md border border-blue-500/10 p-6 rounded-2xl shadow-sm hover:shadow-md transition-all">
            <div className="flex justify-between items-start mb-4">
              <div className="w-12 h-12 bg-emerald-100 dark:bg-emerald-900/30 text-emerald-600 rounded-xl flex items-center justify-center">
                <BrainCircuit className="w-6 h-6" />
              </div>
              <div className="flex -space-x-2">
                <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=C" className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=D" className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=E" className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
              </div>
            </div>
            <h4 className="text-lg font-bold mb-1">AI 研发小组</h4>
            <p className="text-sm text-slate-500 mb-4 truncate">专注大模型算法优化与 API 架构讨论。</p>
            <div className="flex items-center justify-between text-xs font-medium">
              <span className="text-slate-400">25 个相关会议</span>
              <button className="text-blue-600 hover:underline">管理小组成员</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
