import React, { useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { 
  Calendar, 
  Gavel, 
  Verified, 
  Lightbulb, 
  CalendarCheck, 
  Zap, 
  Palette, 
  Bot, 
  CheckCircle2, 
  Plus, 
  RefreshCw, 
  Edit3, 
  MessageSquare, 
  Mail, 
  Download, 
  Star, 
  StarHalf, 
  Play, 
  RotateCcw, 
  RotateCw,
  BarChart
} from 'lucide-react';
import { useMeetings } from '../contexts/MeetingContext';

export function MeetingSummaryDetail() {
  const { id } = useParams<{ id: string }>();
  const { meetings, addKnowledgeBaseEntry } = useMeetings();
  
  const meeting = useMemo(() => meetings.find(m => m.id === id), [meetings, id]);

  const [isEditing, setIsEditing] = React.useState(false);
  const [showDownloadModal, setShowDownloadModal] = React.useState(false);
  const [saveOption, setSaveOption] = React.useState<'local' | 'knowledge' | null>(null);
  const [selectedContent, setSelectedContent] = React.useState({ summary: false, audio: false, chat: false });
  const [audioProgress, setAudioProgress] = React.useState(33); // 0-100
  const [showContactModal, setShowContactModal] = React.useState(false);
  const [contactMethod, setContactMethod] = React.useState<'Slack' | 'Email' | null>(null);
  
  const [summary, setSummary] = React.useState(meeting?.summary || "");
  const [decisions, setDecisions] = React.useState(meeting?.decisions || []);
  const [takeaways, setTakeaways] = React.useState(meeting?.takeaways || []);
  const [actionItems, setActionItems] = React.useState(meeting?.actionItems || []);

  React.useEffect(() => {
    if (meeting) {
      setSummary(meeting.summary);
      setDecisions(meeting.decisions);
      setTakeaways(meeting.takeaways);
      setActionItems(meeting.actionItems);
    }
  }, [meeting]);

  if (!meeting) {
    return <div className="p-8">会议未找到</div>;
  }

  const handleSave = () => {
    setIsEditing(false);
  };

  const handleAddTask = () => {
    const newTask = {
      id: Date.now().toString(),
      task: '新任务',
      status: '未开始',
      assignee: '待分配',
      dueDate: '2023-10-25'
    };
    setActionItems([...actionItems, newTask]);
    // Removed setIsEditing(true) to not auto-enter edit mode
  };

  const handleDeleteTask = (id: string) => {
    setActionItems(actionItems.filter(item => item.id !== id));
  };

  const handleRegenerate = () => {
    setSummary("（AI 已重新生成）本次会议主要讨论了产品迭代计划。团队确认了 V2.1 版本的发布时间，并针对仪表盘加载性能、移动端 UI 设计以及 AI 增强功能进行了深入探讨，达成了多项关键决策与待办事项。");
    setDecisions([
      { id: '1', title: '通过：采用混合云部署架构（已更新）', text: '针对大客户版本采用混合云部署，以满足数据合规性要求并确保性能。' },
      { id: '2', title: '通过：取消旧版 Webhook 维护（已更新）', text: '从 V2.1 版本起全面停止对 v1.0 接口的技术支持，发布迁移指南。' },
    ]);
  };

  const handleConfirmSave = () => {
    if (!saveOption) return;
    
    if (saveOption === 'knowledge') {
      addKnowledgeBaseEntry({
        id: Date.now().toString(),
        meetingId: meeting.id,
        title: meeting.title,
        date: '2023-10-24', // Should probably be part of meeting data
        ...selectedContent
      });
    } else {
      console.log('Saving to local:', selectedContent);
    }
    setShowDownloadModal(false);
  };

  return (
    <div className="mx-auto flex w-full max-w-[1440px] flex-1 gap-10 px-8 py-10">
      <div className="flex flex-1 flex-col gap-8">
        {/* Header */}
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center rounded-md bg-blue-50 dark:bg-blue-500/10 px-2 py-1 text-xs font-bold text-blue-600 dark:text-blue-400 ring-1 ring-inset ring-blue-700/10 dark:ring-blue-500/20">产品迭代</span>
            <span className="text-sm font-medium text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
              <Calendar className="w-4 h-4" />
              {meeting.startTime}
            </span>
          </div>
          <h1 className="text-4xl font-black tracking-tight text-slate-900 dark:text-white">{meeting.title}</h1>
          
          <div className="flex items-center justify-between rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="flex -space-x-3">
                {(meeting.participantNames || []).slice(0, 4).map((name, i) => (
                  <img key={i} src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${name}`} className="h-10 w-10 rounded-full border-2 border-white dark:border-slate-900 bg-slate-100" />
                ))}
                {(meeting.participantNames || []).length > 4 && (
                  <div className="flex h-10 w-10 items-center justify-center rounded-full border-2 border-white dark:border-slate-900 bg-slate-100 dark:bg-slate-800 text-xs font-bold text-slate-600 dark:text-slate-300">+{(meeting.participantNames || []).length - 4}</div>
                )}
              </div>
              <div className="text-sm">
                <p className="font-bold text-slate-900 dark:text-white">参会人员 ({meeting.participants}人)</p>
                <p className="text-slate-500 dark:text-slate-400">{(meeting.participantNames || []).join(', ')}</p>
              </div>
            </div>
            <button className="text-sm font-semibold text-blue-600 dark:text-blue-400 hover:underline">查看签到详情</button>
          </div>
        </div>

        {/* Decisions */}
        <section className="rounded-2xl border-2 border-blue-600 bg-blue-50/50 dark:bg-blue-900/10 p-8">
          <div className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-600/20">
                <Gavel className="w-5 h-5" />
              </div>
              <h2 className="text-2xl font-black text-slate-900 dark:text-white">关键决策 (Decisions)</h2>
            </div>
            <span className="text-xs font-bold uppercase tracking-widest text-blue-600/60 dark:text-blue-400/60">Final Agreements</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {decisions.map((decision, index) => (
              <div key={decision.id} className="flex items-start gap-4 rounded-xl bg-white dark:bg-slate-900 p-5 shadow-sm border border-blue-600/10 dark:border-blue-500/20">
                <Verified className="text-blue-600 dark:text-blue-400 w-6 h-6 shrink-0" />
                <div className="w-full">
                  {isEditing ? (
                    <>
                      <input 
                        value={decision.title}
                        onChange={(e) => {
                          const newDecisions = [...decisions];
                          newDecisions[index].title = e.target.value;
                          setDecisions(newDecisions);
                        }}
                        className="w-full text-lg font-bold text-slate-900 dark:text-white p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800"
                      />
                      <textarea 
                        value={decision.text}
                        onChange={(e) => {
                          const newDecisions = [...decisions];
                          newDecisions[index].text = e.target.value;
                          setDecisions(newDecisions);
                        }}
                        className="w-full mt-1 p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm leading-relaxed text-slate-600 dark:text-slate-400 font-medium"
                        rows={2}
                      />
                    </>
                  ) : (
                    <>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">{decision.title}</p>
                      <p className="mt-1 text-sm leading-relaxed text-slate-600 dark:text-slate-400 font-medium">{decision.text}</p>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Meeting Summary */}
        <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8 shadow-sm">
          <div className="mb-8 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-100 dark:bg-amber-500/20 text-amber-600 dark:text-amber-400">
              <Lightbulb className="w-6 h-6" />
            </div>
            <h2 className="text-2xl font-black text-slate-900 dark:text-white">会议总结 (Meeting Summary)</h2>
          </div>
          
          <div className="space-y-8">
            {/* Summary */}
            <div>
              <h4 className="font-bold text-slate-900 dark:text-white mb-3">会议摘要</h4>
              {isEditing ? (
                <textarea 
                  value={summary}
                  onChange={(e) => setSummary(e.target.value)}
                  className="w-full p-3 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200"
                  rows={4}
                />
              ) : (
                <p className="text-slate-600 dark:text-slate-400 leading-relaxed font-medium">
                  {summary}
                </p>
              )}
            </div>

            {/* Key Takeaways */}
            <div>
              <h4 className="font-bold text-slate-900 dark:text-white mb-4">核心要点 (Key Takeaways)</h4>
              <div className="grid grid-cols-1 gap-6">
                {takeaways.map((item, index) => (
                  <div key={item.id} className={`flex gap-5 ${index > 0 ? 'border-t border-slate-50 dark:border-slate-800 pt-6' : ''}`}>
                    <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-50 dark:bg-blue-500/10 text-blue-600 dark:text-blue-400">
                      <CalendarCheck className="w-5 h-5" />
                    </div>
                    <div className="w-full">
                      {isEditing ? (
                        <>
                          <input 
                            value={item.title}
                            onChange={(e) => {
                              const newTakeaways = [...takeaways];
                              newTakeaways[index].title = e.target.value;
                              setTakeaways(newTakeaways);
                            }}
                            className="w-full font-bold text-slate-900 dark:text-white mb-1 p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800"
                          />
                          <textarea 
                            value={item.text}
                            onChange={(e) => {
                              const newTakeaways = [...takeaways];
                              newTakeaways[index].text = e.target.value;
                              setTakeaways(newTakeaways);
                            }}
                            className="w-full p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400"
                            rows={2}
                          />
                        </>
                      ) : (
                        <>
                          <h4 className="font-bold text-slate-900 dark:text-white">{item.title}</h4>
                          <p className="mt-1 text-slate-600 dark:text-slate-400 leading-relaxed font-medium">{item.text}</p>
                        </>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Action Items */}
        <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 shadow-sm overflow-hidden">
          <div className="p-8 pb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
                <h2 className="text-2xl font-black text-slate-900 dark:text-white">待办事项 (Action Items)</h2>
              </div>
              <button onClick={handleAddTask} className="flex items-center gap-1.5 text-sm font-bold text-blue-600 dark:text-blue-400">
                <Plus className="w-5 h-5" />
                添加任务
              </button>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="bg-slate-50/50 dark:bg-slate-800/50 border-y border-slate-100 dark:border-slate-800">
                  <th className="px-8 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">任务内容</th>
                  <th className="px-8 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 text-center">状态</th>
                  <th className="px-8 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400">负责人</th>
                  <th className="px-8 py-4 text-xs font-bold uppercase tracking-wider text-slate-500 dark:text-slate-400 text-right">截止日期</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {actionItems.map((item, index) => (
                  <tr key={item.id} className="group hover:bg-slate-50/50 dark:hover:bg-slate-800/50 transition-colors">
                    <td className="px-8 py-5 text-sm font-bold text-slate-900 dark:text-white">
                      {isEditing ? (
                        <input 
                          value={item.task}
                          onChange={(e) => {
                            const newActionItems = [...actionItems];
                            newActionItems[index].task = e.target.value;
                            setActionItems(newActionItems);
                          }}
                          className="w-full p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800"
                        />
                      ) : item.task}
                    </td>
                    <td className="px-8 py-5 text-center">
                      {isEditing ? (
                        <input 
                          value={item.status}
                          onChange={(e) => {
                            const newActionItems = [...actionItems];
                            newActionItems[index].status = e.target.value;
                            setActionItems(newActionItems);
                          }}
                          className="w-full p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-center"
                        />
                      ) : (
                        <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-[11px] font-black ${item.status === '已完成' ? 'bg-emerald-100 dark:bg-emerald-500/20 text-emerald-700 dark:text-emerald-400' : item.status === '进行中' ? 'bg-blue-100 dark:bg-blue-500/20 text-blue-700 dark:text-blue-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'}`}>
                          {item.status}
                        </span>
                      )}
                    </td>
                    <td className="px-8 py-5">
                      <div className="flex items-center gap-2.5">
                        <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=Assignee${index + 1}`} className="h-7 w-7 rounded-full shadow-sm bg-slate-100" />
                        {isEditing ? (
                          <input 
                            value={item.assignee}
                            onChange={(e) => {
                              const newActionItems = [...actionItems];
                              newActionItems[index].assignee = e.target.value;
                              setActionItems(newActionItems);
                            }}
                            className="w-full p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800"
                          />
                        ) : (
                          <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">{item.assignee}</span>
                        )}
                      </div>
                    </td>
                    <td className="px-8 py-5 text-right text-sm font-medium text-slate-500 dark:text-slate-400">
                      <div className="flex items-center justify-end gap-2">
                        {isEditing ? (
                          <input 
                            type="date"
                            value={item.dueDate}
                            onChange={(e) => {
                              const newActionItems = [...actionItems];
                              newActionItems[index].dueDate = e.target.value;
                              setActionItems(newActionItems);
                            }}
                            className="w-full p-2 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-right"
                          />
                        ) : item.dueDate}
                        {isEditing && (
                          <button onClick={() => handleDeleteTask(item.id)} className="text-red-500 hover:text-red-700">
                            删除
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Voting Results */}
        {meeting.votes && meeting.votes.length > 0 && (
          <section className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8 shadow-sm">
            <div className="mb-8 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400">
                <BarChart className="w-6 h-6" />
              </div>
              <h2 className="text-2xl font-black text-slate-900 dark:text-white">投票结果 (Voting Results)</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {meeting.votes.map((vote) => (
                <div key={vote.id} className="rounded-xl border border-slate-100 dark:border-slate-800 p-6 bg-slate-50/50 dark:bg-slate-800/30">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-bold text-lg text-slate-900 dark:text-white">{vote.title}</h4>
                    <span className="text-xs font-bold px-2 py-1 bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 rounded-lg">
                      {vote.totalVotes} 人参与
                    </span>
                  </div>
                  <div className="space-y-4">
                    {vote.options.map((opt) => {
                      const percentage = vote.totalVotes > 0 ? Math.round((opt.votes / vote.totalVotes) * 100) : 0;
                      return (
                        <div key={opt.id} className="space-y-1.5">
                          <div className="flex justify-between text-sm">
                            <span className="font-medium text-slate-700 dark:text-slate-300">{opt.text}</span>
                            <span className="font-bold text-slate-900 dark:text-white">{opt.votes} 票 ({percentage}%)</span>
                          </div>
                          <div className="h-2 w-full bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                            <div 
                              className={`h-full transition-all duration-1000 ${opt.text === vote.winner ? 'bg-blue-600' : 'bg-slate-400 dark:bg-slate-500'}`}
                              style={{ width: `${percentage}%` }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-4 pt-4 border-t border-slate-200 dark:border-slate-700 flex items-center gap-2">
                    <span className="text-xs font-bold text-slate-500 uppercase">获胜项:</span>
                    <span className="text-sm font-black text-blue-600">{vote.winner}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        <div className="flex justify-center pt-4 pb-12">
          <button onClick={handleRegenerate} className="group flex items-center gap-3 rounded-full border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-8 py-3.5 text-sm font-black text-slate-600 dark:text-slate-300 shadow-sm hover:border-blue-600 hover:text-blue-600 dark:hover:border-blue-500 dark:hover:text-blue-400 transition-all active:scale-95">
            <RefreshCw className="w-5 h-5 transition-transform group-hover:rotate-180" />
            重新生成摘要内容
          </button>
        </div>
      </div>

      {/* Right Sidebar */}
      <aside className="w-80 shrink-0">
        <div className="sticky top-[88px] flex flex-col gap-6">
          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6 shadow-sm">
            <button 
              onClick={() => setIsEditing(!isEditing)}
              className="mb-6 flex w-full items-center justify-center gap-3 rounded-xl bg-blue-600 py-4 text-sm font-black text-white shadow-lg shadow-blue-600/30 transition-all hover:translate-y-[-1px] hover:shadow-blue-600/40 active:translate-y-0"
            >
              <Edit3 className="w-5 h-5" />
              {isEditing ? '保存摘要' : '编辑摘要'}
            </button>
            <div className="mb-8">
              <p className="mb-4 text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">快速分享 / Sharing</p>
              <div className="grid grid-cols-3 gap-3">
                <button onClick={() => { setContactMethod('Slack'); setShowContactModal(true); }} className="flex flex-col items-center justify-center gap-2 rounded-xl border border-slate-100 dark:border-slate-800 py-4 hover:border-blue-600/30 hover:bg-blue-50/50 dark:hover:bg-blue-900/20 transition-all">
                  <MessageSquare className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                  <span className="text-[10px] font-bold text-slate-600 dark:text-slate-300">Slack</span>
                </button>
                <button onClick={() => { setContactMethod('Email'); setShowContactModal(true); }} className="flex flex-col items-center justify-center gap-2 rounded-xl border border-slate-100 dark:border-slate-800 py-4 hover:border-blue-600/30 hover:bg-blue-50/50 dark:hover:bg-blue-900/20 transition-all">
                  <Mail className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                  <span className="text-[10px] font-bold text-slate-600 dark:text-slate-300">Email</span>
                </button>
                <button onClick={() => setShowDownloadModal(!showDownloadModal)} className="flex flex-col items-center justify-center gap-2 rounded-xl border border-slate-100 dark:border-slate-800 py-4 hover:border-blue-600/30 hover:bg-blue-50/50 dark:hover:bg-blue-900/20 transition-all">
                  <Download className="w-5 h-5 text-slate-500 dark:text-slate-400" />
                  <span className="text-[10px] font-bold text-slate-600 dark:text-slate-300">Export</span>
                </button>
              </div>
            </div>
            <div className="pt-6 border-t border-slate-100 dark:border-slate-800">
              <p className="mb-3 text-[10px] font-black uppercase tracking-[0.2em] text-slate-400">摘要质量评分</p>
              <div className="flex items-center justify-between">
                <div className="flex gap-0.5 text-amber-400">
                  <Star className="w-5 h-5 fill-current" />
                  <Star className="w-5 h-5 fill-current" />
                  <Star className="w-5 h-5 fill-current" />
                  <Star className="w-5 h-5 fill-current" />
                  <StarHalf className="w-5 h-5 fill-current" />
                </div>
                <span className="text-base font-black text-slate-900 dark:text-white">4.5</span>
              </div>
              <p className="mt-2 text-[11px] font-medium leading-relaxed text-slate-400">您的反馈将直接帮助我们优化针对您团队的训练模型。</p>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/50 p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex h-2 w-2 rounded-full bg-red-500 animate-pulse"></span>
                <span className="text-xs font-black text-slate-900 dark:text-white uppercase">Recording</span>
              </div>
              <span className="font-mono text-[11px] font-bold text-slate-500 dark:text-slate-400">01:24:45</span>
            </div>
            <div className="relative h-1.5 w-full rounded-full bg-slate-300 dark:bg-slate-700 cursor-pointer" onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = e.clientX - rect.left;
                setAudioProgress((x / rect.width) * 100);
              }}>
                <div className="absolute inset-y-0 left-0 h-full rounded-full bg-blue-600" style={{ width: `${audioProgress}%` }}></div>
                <div className="absolute top-1/2 -translate-y-1/2 h-3 w-3 rounded-full border-2 border-white dark:border-slate-900 bg-blue-600 shadow-sm" style={{ left: `${audioProgress}%` }}></div>
              </div>
            <div className="mt-6 flex items-center justify-center gap-6">
              <button className="text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors">
                <RotateCcw className="w-6 h-6" />
              </button>
              <button className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-600 text-white shadow-lg shadow-blue-600/20 hover:scale-105 transition-transform active:scale-95">
                <Play className="w-6 h-6 fill-current" />
              </button>
              <button className="text-slate-400 hover:text-slate-900 dark:hover:text-white transition-colors">
                <RotateCw className="w-6 h-6" />
              </button>
            </div>
          </div>

          {showContactModal && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
              <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8 shadow-2xl w-full max-w-md">
                <div className="flex justify-between items-center mb-6">
                  <h4 className="text-xl font-bold text-slate-900 dark:text-white">分享至 {contactMethod}</h4>
                  <button onClick={() => setShowContactModal(false)} className="text-slate-500 hover:text-slate-900">关闭</button>
                </div>
                <div className="mb-6">
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">选择联系人</label>
                  <select className="w-full p-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
                    <option>Alice (主持人)</option>
                    <option>Bob Chen</option>
                  </select>
                </div>
                <button onClick={() => setShowContactModal(false)} className="w-full p-4 rounded-lg bg-blue-600 text-white font-bold hover:bg-blue-700">发送</button>
              </div>
            </div>
          )}

          {showDownloadModal && (
            /* Download Modal */
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
              <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-8 shadow-2xl w-full max-w-md">
                <div className="flex justify-between items-center mb-6">
                  <h4 className="text-xl font-bold text-slate-900 dark:text-white">保存</h4>
                  <button onClick={() => setShowDownloadModal(false)} className="text-slate-500 hover:text-slate-900">关闭</button>
                </div>
                <div className="space-y-3 mb-6">
                  <label className="flex items-center gap-3 p-4 rounded-lg border border-slate-200 cursor-pointer hover:border-blue-500">
                    <input type="radio" name="saveOption" value="local" checked={saveOption === 'local'} onChange={() => setSaveOption('local')} className="w-4 h-4" />
                    保存到本地 (ZIP)
                  </label>
                  <label className="flex items-center gap-3 p-4 rounded-lg border border-slate-200 cursor-pointer hover:border-blue-500">
                    <input type="radio" name="saveOption" value="knowledge" checked={saveOption === 'knowledge'} onChange={() => setSaveOption('knowledge')} className="w-4 h-4" />
                    保存到知识库
                  </label>
                </div>
                <div className="space-y-3 mb-8">
                  <label className="flex items-center gap-3 text-sm"><input type="checkbox" checked={selectedContent.summary} onChange={(e) => setSelectedContent(prev => ({...prev, summary: e.target.checked}))} className="w-4 h-4" /> 会议总结</label>
                  <label className="flex items-center gap-3 text-sm"><input type="checkbox" checked={selectedContent.audio} onChange={(e) => setSelectedContent(prev => ({...prev, audio: e.target.checked}))} className="w-4 h-4" /> 录像音频</label>
                  <label className="flex items-center gap-3 text-sm"><input type="checkbox" checked={selectedContent.chat} onChange={(e) => setSelectedContent(prev => ({...prev, chat: e.target.checked}))} className="w-4 h-4" /> 聊天消息</label>
                </div>
                <button onClick={handleConfirmSave} disabled={!saveOption} className="w-full p-4 rounded-lg bg-blue-600 text-white font-bold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed">确定</button>
              </div>
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
