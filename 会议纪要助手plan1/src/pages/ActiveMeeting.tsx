import React, { useState, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { useMeetings } from '../contexts/MeetingContext';
import { 
  Mic, 
  MicOff,
  Video, 
  VideoOff,
  Share2, 
  PauseCircle, 
  PlayCircle,
  BarChart, 
  UploadCloud, 
  PhoneOff, 
  Settings, 
  Edit3, 
  PlusCircle, 
  GripVertical, 
  Trash2, 
  Vote,
  X,
  Plus,
  Sparkles,
  FileText,
  ChevronUp,
  ChevronDown,
  Maximize2,
  Minimize2,
  CheckCircle2,
  Loader2,
  Clipboard
} from 'lucide-react';
import { TranscriptsArea } from '../components/TranscriptsArea';

export function ActiveMeeting() {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const { meetings, endMeeting } = useMeetings();
  const meeting = (meetings || []).find(m => m.id === id);
  const [todos, setTodos] = useState<any[]>([]);
  const [isAddingTodo, setIsAddingTodo] = useState(false);
  const [isLoadingTodos, setIsLoadingTodos] = useState(false);
  const [newTodo, setNewTodo] = useState('');
  const [isMicOn, setIsMicOn] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isVideoExpanded, setIsVideoExpanded] = useState(true);
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(true);
  const [isTodoExpanded, setIsTodoExpanded] = useState(true);
  const [isVoteExpanded, setIsVoteExpanded] = useState(true);
  const [isFullScreen, setIsFullScreen] = useState(false);
  const [maximizedVideo, setMaximizedVideo] = useState<string | null>(null);
  const [voteOptions, setVoteOptions] = useState([{ id: '1', text: '' }, { id: '2', text: '' }]);
  const [voteTitle, setVoteTitle] = useState('');
  const [activeVote, setActiveVote] = useState<{ 
    title: string, 
    options: { id: string, text: string, votes: number }[],
    expiresAt: number 
  } | null>(null);
  const [voteTimeLeft, setVoteTimeLeft] = useState<number>(0);
  const [voteDuration, setVoteDuration] = useState<string>("1");
  const [showVoteNotification, setShowVoteNotification] = useState(false);
  const [pendingVote, setPendingVote] = useState<{ title: string, options: { id: string, text: string }[], duration: number } | null>(null);
  const [chatMessages, setChatMessages] = useState([
    { id: '1', sender: '张伟', role: '主持人', text: '大家好，今天我们的会议主要讨论 Q4 季度的市场推广计划。李芳，你那边关于社交媒体渠道的预算初稿准备好了吗？', time: '10:05:22', isUser: false },
    { id: '2', sender: '李芳', role: '产品经理', text: '是的，张伟。我已经整理好了。初步预算我们打算向短视频平台倾斜 40% 的资源，尤其是针对年轻人群体的投放。我们需要在下周三前确定最终的供应商名单。', time: '10:05:45', isUser: false },
  ]);
  const [chatMessage, setChatMessage] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const [selectedVoteResult, setSelectedVoteResult] = useState<{ title: string, options: { id: string, text: string, votes: number }[] } | null>(null);
  const [userVoteSelection, setUserVoteSelection] = useState<string | null>(null);
  const [showSummaryModal, setShowSummaryModal] = useState(false);
  const [summaryData, setSummaryData] = useState<{ decisions: string[], actions: string[] } | null>(null);
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);

  // Fetch Todos Effect
  React.useEffect(() => {
    const fetchTodos = async () => {
      setIsLoadingTodos(true);
      try {
        const res = await fetch(`/api/todos/${id}`);
        if (!res.ok) throw new Error('获取待办失败');
        const data = await res.json();
        setTodos(data);
      } catch (error: any) {
        console.error(error);
      } finally {
        setIsLoadingTodos(false);
      }
    };
    if (id) fetchTodos();
  }, [id]);

  // Sync Effect
  React.useEffect(() => {
    const syncInterval = setInterval(async () => {
      setIsSyncing(true);
      try {
        const response = await fetch('/api/sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ meetingId: id })
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const text = await response.text();
        const data = text ? JSON.parse(text) : {};
        
        if (data.todos) setTodos(data.todos);
        // Do not let a null activeVote from the server override an active local poll.
        if (data.activeVote !== undefined && data.activeVote !== null) setActiveVote(data.activeVote);
      } catch (error) {
        console.error('Sync failed', error);
      } finally {
        setIsSyncing(false);
      }
    }, 5000);
    return () => clearInterval(syncInterval);
  }, [id]);

  // Vote Countdown Effect
  React.useEffect(() => {
    let timer: NodeJS.Timeout;
    if (activeVote && voteTimeLeft > 0) {
      timer = setInterval(() => {
        setVoteTimeLeft(prev => {
          if (prev <= 1) {
            // Vote expired
            handleSubmitVote();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [activeVote, voteTimeLeft]);

  // Background Vote Simulation Effect
  React.useEffect(() => {
    let timer: NodeJS.Timeout;
    if (activeVote && voteTimeLeft > 0) {
      timer = setInterval(() => {
        // Randomly add a vote from "others" every 3-7 seconds
        const randomIdx = Math.floor(Math.random() * activeVote.options.length);
        setActiveVote(prev => {
          if (!prev) return null;
          const newOptions = [...prev.options];
          newOptions[randomIdx] = { ...newOptions[randomIdx], votes: newOptions[randomIdx].votes + 1 };
          return { ...prev, options: newOptions };
        });
      }, 3000 + Math.random() * 4000);
    }
    return () => clearInterval(timer);
  }, [activeVote, !!activeVote]);

  const handleSendMessage = () => {
    if (chatMessage.trim()) {
      setChatMessages([...chatMessages, {
        id: Date.now().toString(),
        sender: '张经理',
        role: '主持人',
        text: chatMessage,
        time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        isUser: true
      }]);
      setChatMessage('');
    }
  };

  const handleTranscript = (text: string) => {
    if (text.trim()) {
      setChatMessages(prev => [...prev, {
        id: Date.now().toString(),
        sender: '我 (语音识别)',
        role: '主持人',
        text,
        time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        isUser: true
      }]);
    }
  };

  const participants = [
    { id: '1', name: '张伟', role: '主持人', initial: '张', color: 'bg-blue-600' },
    { id: '2', name: '李芳', role: '产品经理', initial: '李', color: 'bg-emerald-600' },
    { id: '3', name: '王强', role: '技术总监', initial: '王', color: 'bg-purple-600' },
    { id: '4', name: '赵敏', role: '设计师', initial: '赵', color: 'bg-amber-600' },
  ];

  const toggleMic = async () => {
    if (!isMicOn) {
      try {
        await navigator.mediaDevices.getUserMedia({ audio: true });
        setIsMicOn(true);
      } catch (err) {
        alert('无法获取麦克风权限，请检查浏览器设置。');
      }
    } else {
      setIsMicOn(false);
    }
  };

  const toggleCamera = async () => {
    if (!isCameraOn) {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        setIsCameraOn(true);
      } catch (err) {
        alert('无法获取摄像头权限，请检查浏览器设置。');
      }
    } else {
      setIsCameraOn(false);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      alert(`已选择文件: ${file.name}，正在上传...`);
      // In a real app, you would handle the file upload to a server here
    }
  };

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullScreen(true);
    } else {
      document.exitFullscreen();
      setIsFullScreen(false);
    }
  };

  const handleAddTodo = async () => {
    if (!newTodo.trim() || isAddingTodo) return;
    setIsAddingTodo(true);
    try {
      const taskText = newTodo.trim();
      const dueDate = new Date().toISOString().split('T')[0];
      const res = await fetch('/api/todos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          meetingId: id,
          task: taskText,
          status: '进行中',
          assignee: '待分配',
          dueDate
        })
      });

      if (!res.ok) throw new Error('发布待办失败');
      
      const { id: newId } = await res.json();
      setTodos([...todos, { 
        id: newId, 
        text: taskText, 
        checked: false, 
        status: '进行中', 
        assignee: '待分配', 
        dueDate 
      }]);
      setNewTodo('');
    } catch (error: any) {
      console.error('[handleAddTodo]', error);
      alert(error.message);
    } finally {
      setIsAddingTodo(false);
    }
  };

  const handleUpdateTodo = async (id: string, updates: any) => {
    try {
      const res = await fetch(`/api/todos/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      });
      if (!res.ok) throw new Error('更新待办失败');
    } catch (error: any) {
      console.error('[handleUpdateTodo]', error);
      alert(error.message);
    }
  };

  const handleDeleteTodo = async (id: string) => {
    try {
      const res = await fetch(`/api/todos/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('删除待办失败');
      setTodos(todos.filter(t => t.id !== id));
    } catch (error: any) {
      console.error('[handleDeleteTodo]', error);
      alert(error.message);
    }
  };

  const handleAddVoteOption = () => {
    setVoteOptions([...voteOptions, { id: Date.now().toString(), text: '' }]);
  };

  const handleRemoveVoteOption = (id: string) => {
    if (voteOptions.length > 2) {
      setVoteOptions(voteOptions.filter(opt => opt.id !== id));
    }
  };

  const handleVoteOptionChange = (id: string, text: string) => {
    setVoteOptions(voteOptions.map(opt => opt.id === id ? { ...opt, text } : opt));
  };

  const handleStartVote = () => {
    if (voteTitle.trim() && voteOptions.filter(opt => opt.text.trim()).length >= 2) {
      setPendingVote({
        title: voteTitle,
        options: voteOptions.filter(opt => opt.text.trim()),
        duration: parseInt(voteDuration) * 60
      });
      setShowVoteNotification(true);
      setVoteTitle('');
      setVoteOptions([{ id: '1', text: '' }, { id: '2', text: '' }]);
    } else {
      alert('请填写投票标题和至少两个选项');
    }
  };

  const handleAcceptVote = () => {
    if (pendingVote) {
      setActiveVote({
        title: pendingVote.title,
        options: (pendingVote.options || []).map(opt => ({ ...opt, votes: 0 })),
        expiresAt: Date.now() + pendingVote.duration * 1000
      });
      setVoteTimeLeft(pendingVote.duration);
      setUserVoteSelection(null);
      setPendingVote(null);
      setShowVoteNotification(false);
    }
  };

  const handleCastVote = (optionId: string) => {
    if (activeVote) {
      setUserVoteSelection(optionId);
    }
  };

  const handleSubmitVote = () => {
    if (activeVote) {
      // Finalize the user's vote if they selected something
      let finalOptions = [...activeVote.options];
      if (userVoteSelection) {
        finalOptions = finalOptions.map(opt => 
          opt.id === userVoteSelection ? { ...opt, votes: opt.votes + 1 } : opt
        );
      }

      const totalVotes = finalOptions.reduce((sum, opt) => sum + opt.votes, 0);
      const winner = [...finalOptions].sort((a, b) => b.votes - a.votes)[0];
      
      // Add vote result to chat
      setChatMessages(prev => [...prev, {
        id: Date.now().toString(),
        sender: '系统助手',
        role: '助手',
        text: `投票结束！"${activeVote.title}" 共有 ${totalVotes} 人参与投票。最终结果：${winner.text} (${winner.votes}票)`,
        time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        isUser: false,
        voteData: {
          title: activeVote.title,
          options: finalOptions
        }
      }]);

      setActiveVote(null);
      setVoteTimeLeft(0);
      setUserVoteSelection(null);
    }
  };

  const handleEndMeeting = async () => {
    setShowSummaryModal(true);
    setIsGeneratingSummary(true);
    try {
      const response = await fetch('/api/generate-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chatMessages, meetingId: id, title: meeting?.title })
      });

      if (!response.ok) throw new Error('总结生成失败');
      const data = await response.json();
      
      if (!data || (!data.decisions && !data.actions)) throw new Error('总结生成失败');

      setSummaryData({
        decisions: data.decisions || [],
        actions: data.actions || []
      });

      // Save to knowledge base
      await fetch('/api/knowledge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          meetingId: id,
          title: (meeting?.title || '会议') + ' 总结',
          summary: (data.decisions || []).join('\n'),
          actionItems: data.actions || []
        })
      });

    } catch (error: any) {
      alert(error.message || '总结生成失败');
    } finally {
      setIsGeneratingSummary(false);
    }
  };

  const confirmEndMeeting = () => {
    // Map todos to ActionItem format
    const finalActionItems = todos.map(todo => ({
      id: todo.id,
      task: todo.text,
      status: todo.checked ? '已完成' : '进行中',
      assignee: '待分配',
      dueDate: new Date().toISOString().split('T')[0]
    }));

    // Collect voting data from chat messages
    const finalVotes = chatMessages
      .filter(msg => msg.voteData)
      .map((msg, idx) => {
        const totalVotes = msg.voteData!.options.reduce((sum, o) => sum + o.votes, 0);
        const winner = [...msg.voteData!.options].sort((a, b) => b.votes - a.votes)[0];
        return {
          id: `v-${idx}-${Date.now()}`,
          title: msg.voteData!.title,
          options: msg.voteData!.options,
          totalVotes,
          winner: winner.text
        };
      });
    
    endMeeting(id!, finalActionItems, finalVotes);
    navigate(`/meetings/${id}`);
  };

  return (
    <div className={`flex flex-col bg-slate-50 dark:bg-slate-950 text-sm transition-all duration-300 ${isFullScreen ? 'fixed inset-0 z-50' : 'h-[calc(100vh-4rem)]'}`}>
      {/* Header */}
      <header className="flex items-center justify-between border-b border-blue-500/20 bg-white dark:bg-slate-900 px-6 py-3 shrink-0">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Mic className="text-blue-600 w-6 h-6" />
            <h1 className="text-xl font-bold tracking-tight">{meeting?.title || 'AI 会议主持人管理界面'}</h1>
          </div>
          <div className="h-6 w-px bg-blue-500/20"></div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-blue-500/10 px-3 py-1.5 rounded-lg border border-blue-500/20">
              <span className="text-lg font-mono font-bold text-blue-600">00:24:15</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2.5 h-2.5 rounded-full ${isPaused ? 'bg-amber-500' : 'bg-red-500 animate-pulse'}`}></span>
              <span className={`text-sm font-medium ${isPaused ? 'text-amber-500' : 'text-red-500'}`}>
                {isPaused ? '已暂停录制' : '正在录制 (主持人模式)'}
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button onClick={toggleFullScreen} className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors">
            {isFullScreen ? <Minimize2 className="w-5 h-5 text-slate-600" /> : <Maximize2 className="w-5 h-5 text-slate-600" />}
          </button>
          <div className="flex -space-x-2">
            <div className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-blue-600 flex items-center justify-center text-xs font-bold text-white">张</div>
            <div className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-emerald-600 flex items-center justify-center text-xs font-bold text-white">李</div>
            <div className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-purple-600 flex items-center justify-center text-xs font-bold text-white">王</div>
            <div className="w-8 h-8 rounded-full border-2 border-white dark:border-slate-900 bg-slate-700 flex items-center justify-center text-xs font-bold text-white">+5</div>
          </div>
          <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500/10 hover:bg-blue-500/20 text-blue-600 transition-colors font-medium">
            <Share2 className="w-4 h-4" />
            <span>邀请参会</span>
          </button>
        </div>
      </header>

      <main className="flex flex-1 overflow-hidden">
        {/* Left Area: Video Grid + Transcription */}
        <section className="flex flex-1 flex-col bg-white dark:bg-slate-900 relative">
          
          {/* Video Grid */}
          <div className={`relative transition-all duration-300 ease-in-out bg-slate-900 flex flex-col ${isVideoExpanded ? (maximizedVideo ? 'h-[32rem]' : 'h-64') : 'h-0'}`}>
            <div className="flex-1 overflow-hidden p-4">
              {maximizedVideo ? (
                <div className="w-full h-full relative rounded-xl overflow-hidden bg-slate-800 border border-slate-700">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className={`w-24 h-24 rounded-full ${participants.find(p => p.id === maximizedVideo)?.color} flex items-center justify-center text-4xl text-white font-bold`}>
                      {participants.find(p => p.id === maximizedVideo)?.initial}
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-lg text-white text-sm flex items-center gap-2">
                    <Mic className="w-4 h-4" />
                    {participants.find(p => p.id === maximizedVideo)?.name}
                  </div>
                  <button 
                    onClick={() => setMaximizedVideo(null)}
                    className="absolute top-4 right-4 p-2 bg-black/50 hover:bg-black/70 rounded-lg text-white transition-colors"
                  >
                    <Minimize2 className="w-5 h-5" />
                  </button>
                </div>
              ) : (
                <div className="w-full h-full grid grid-cols-2 md:grid-cols-4 gap-4">
                  {participants.map(p => (
                    <div key={p.id} className="relative rounded-xl overflow-hidden bg-slate-800 border border-slate-700 group">
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className={`w-16 h-16 rounded-full ${p.color} flex items-center justify-center text-2xl text-white font-bold`}>
                          {p.initial}
                        </div>
                      </div>
                      <div className="absolute bottom-3 left-3 bg-black/50 backdrop-blur-md px-2 py-1 rounded text-white text-xs flex items-center gap-1.5">
                        <Mic className="w-3 h-3" />
                        {p.name}
                      </div>
                      <button 
                        onClick={() => setMaximizedVideo(p.id)}
                        className="absolute top-3 right-3 p-1.5 bg-black/50 hover:bg-black/70 rounded text-white opacity-0 group-hover:opacity-100 transition-all"
                      >
                        <Maximize2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          {/* 语音弹幕组件 */}
          <div className="px-4 pt-4">
             <TranscriptsArea isMicOn={isMicOn} onTranscript={handleTranscript} />
          </div>

          {/* Toggle Button */}
          <div className="relative h-0 z-10 flex justify-center">
             <button 
               onClick={() => setIsVideoExpanded(!isVideoExpanded)} 
               className="absolute -top-0 bg-white dark:bg-slate-800 border-x border-b border-slate-200 dark:border-slate-700 rounded-b-xl px-6 py-1 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-400 hover:text-blue-600"
             >
                {isVideoExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
             </button>
          </div>

          {/* Vote Notification Modal */}
          {showVoteNotification && pendingVote && (
            <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 z-50 bg-white dark:bg-slate-800 p-4 rounded-xl shadow-2xl border border-blue-500/30 w-80 cursor-pointer hover:border-blue-500 transition-colors" onClick={handleAcceptVote}>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center">
                  <Vote className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
                <div>
                  <h4 className="font-bold text-slate-900 dark:text-slate-100">新投票提醒</h4>
                  <p className="text-xs text-slate-500">点击查看并参与投票: {pendingVote.title}</p>
                </div>
              </div>
            </div>
          )}

          {/* Active Vote Modal */}
          {activeVote && (
            <div className="absolute inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
              <div className="bg-white dark:bg-slate-900 rounded-2xl p-6 w-full max-w-sm shadow-2xl border border-slate-200 dark:border-slate-700">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Vote className="w-5 h-5 text-blue-600" />
                    <h3 className="font-bold text-slate-900 dark:text-slate-100">{activeVote.title}</h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono font-bold text-red-500 bg-red-50 dark:bg-red-500/10 px-2 py-1 rounded">
                      {Math.floor(voteTimeLeft / 60)}:{(voteTimeLeft % 60).toString().padStart(2, '0')}
                    </span>
                    <button onClick={() => setActiveVote(null)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                <div className="space-y-2 mb-6">
                  {activeVote.options.map((opt) => {
                    const isSelected = userVoteSelection === opt.id;
                    return (
                      <button 
                        key={opt.id} 
                        onClick={() => handleCastVote(opt.id)}
                        className={`w-full relative overflow-hidden group px-4 py-3 rounded-lg border transition-all text-sm ${
                          isSelected 
                            ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20 text-blue-600' 
                            : 'border-slate-200 dark:border-slate-700 hover:border-blue-500 text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        <div className="relative flex justify-between items-center">
                          <span className={isSelected ? 'font-bold' : ''}>{opt.text}</span>
                          {isSelected ? (
                            <CheckCircle2 className="w-4 h-4" />
                          ) : (
                            <span className="text-xs text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">点击选择</span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
                <button 
                  onClick={handleSubmitVote} 
                  disabled={!userVoteSelection}
                  className={`w-full py-2.5 rounded-xl text-sm font-bold text-white transition-all ${
                    userVoteSelection 
                      ? 'bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/20' 
                      : 'bg-slate-300 cursor-not-allowed'
                  }`}
                >
                  确认并提交投票
                </button>
              </div>
            </div>
          )}

          {/* Transcription Area */}
          <div className="flex-1 overflow-y-auto p-6 relative">
            {/* Active Vote Overlay */}
            {activeVote && (
              <div className="sticky top-0 z-20 mb-6 bg-white dark:bg-slate-800 border border-blue-200 dark:border-blue-500/30 shadow-lg rounded-xl p-4 w-full max-w-2xl mx-auto">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Vote className="w-5 h-5 text-blue-600" />
                    <h3 className="font-bold text-slate-900 dark:text-slate-100">{activeVote.title}</h3>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-mono font-bold text-red-500 bg-red-50 dark:bg-red-500/10 px-2 py-1 rounded">
                      倒计时 {Math.floor(voteTimeLeft / 60)}:{(voteTimeLeft % 60).toString().padStart(2, '0')}
                    </span>
                    <button onClick={() => setActiveVote(null)} className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {activeVote.options.map((opt) => {
                    const isSelected = userVoteSelection === opt.id;
                    return (
                      <button 
                        key={opt.id} 
                        onClick={() => handleCastVote(opt.id)}
                        className={`relative overflow-hidden group px-4 py-3 rounded-lg border transition-all text-sm ${
                          isSelected 
                            ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20 text-blue-600' 
                            : 'border-slate-200 dark:border-slate-700 hover:border-blue-500 text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        <div className="relative flex justify-between items-center">
                          <span className={`font-medium ${isSelected ? 'text-blue-600' : ''}`}>{opt.text}</span>
                          {isSelected ? (
                            <CheckCircle2 className="w-4 h-4" />
                          ) : (
                            <span className="text-[10px] text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">点击选择</span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="max-w-4xl mx-auto w-full space-y-8">
              {chatMessages.map((msg) => (
                <div key={msg.id} className={`flex gap-4 ${msg.isUser ? 'flex-row-reverse' : ''}`}>
                  <div className="flex-shrink-0">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center font-bold text-white ${msg.isUser ? 'bg-slate-700' : (msg.sender === '张伟' ? 'bg-blue-600' : 'bg-emerald-600')}`}>
                      {(msg.sender || '').slice(0, 2)}
                    </div>
                  </div>
                  <div className={`flex flex-col gap-1 ${msg.isUser ? 'items-end' : ''}`}>
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-semibold text-slate-700 dark:text-slate-300">{msg.sender}</span>
                      <span className="text-xs text-slate-500 tracking-wider">{msg.time}</span>
                    </div>
                    <div className={`text-lg leading-relaxed p-4 rounded-xl shadow-sm border ${msg.isUser ? 'bg-blue-600 text-white rounded-tr-none' : 'bg-white dark:bg-blue-900/40 text-slate-800 dark:text-blue-50 rounded-tl-none border-blue-200 dark:border-blue-500/30'}`}>
                      {msg.text}
                      {msg.voteData && (
                        <div className="mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
                          <button 
                            onClick={() => setSelectedVoteResult(msg.voteData)}
                            className="flex items-center gap-2 text-sm font-bold text-blue-600 dark:text-blue-400 hover:underline"
                          >
                            <BarChart className="w-4 h-4" />
                            查看投票详情
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-lg bg-emerald-600 flex items-center justify-center font-bold text-white">李芳</div>
              </div>
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">李芳</span>
                  <span className="text-xs text-slate-500 tracking-wider">10:05:45</span>
                </div>
                <p className="text-lg leading-relaxed text-slate-800 dark:text-emerald-50 bg-white dark:bg-emerald-900/40 p-4 rounded-xl rounded-tl-none border border-emerald-200 dark:border-emerald-500/30 shadow-sm">
                  是的，张伟。我已经整理好了。初步预算我们打算向短视频平台倾斜 40% 的资源，尤其是针对年轻用户群体的投放。我们需要在下周三前确定最终的供应商名单。
                </p>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-lg bg-purple-600 flex items-center justify-center font-bold text-white">王强</div>
              </div>
              <div className="flex flex-col gap-1 w-full">
                <div className="flex items-center gap-3">
                  <span className="text-sm font-semibold text-purple-600 dark:text-purple-400">王强</span>
                  <span className="text-xs text-slate-500 tracking-wider">正在发言...</span>
                </div>
                <div className="relative bg-white dark:bg-purple-900/40 p-4 rounded-xl rounded-tl-none border border-purple-200 dark:border-purple-500/30 shadow-sm">
                  <p className="text-lg leading-relaxed text-slate-800 dark:text-purple-50 italic">
                    关于供应商的资质审核，我认为我们需要更严格的 KPI 考核体系，比如转化率和用户留存...
                  </p>
                  <div className="absolute -bottom-1 left-0 w-full h-0.5 bg-gradient-to-r from-purple-500/50 to-transparent"></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Host Controls Sidebar */}
        <aside className="w-96 border-l border-blue-500/20 bg-white dark:bg-slate-900 p-5 flex flex-col gap-6 overflow-y-auto shrink-0">
          <div className="flex items-center justify-between border-b border-blue-500/10 pb-4">
            <div className="flex items-center gap-2">
              <Settings className="text-blue-600 w-5 h-5" />
              <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">主持人控制面板</h2>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-500">同步到全体人员</span>
              <button 
                onClick={() => setIsSidebarExpanded(!isSidebarExpanded)}
                className={`relative inline-flex h-5 w-10 shrink-0 cursor-pointer items-center rounded-full transition-colors focus:outline-none ${isSidebarExpanded ? 'bg-blue-600' : 'bg-slate-300'}`}
              >
                <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${isSidebarExpanded ? 'translate-x-5' : 'translate-x-1'}`}></span>
              </button>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Edit3 className="text-blue-600 w-5 h-5" />
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">实时编辑待办</h3>
                {isSyncing && (
                  <div className="flex items-center gap-1 text-xs text-blue-600">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    <span>同步中...</span>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2">
                <button onClick={() => setIsTodoExpanded(!isTodoExpanded)} className="text-slate-400 hover:text-slate-600">
                  <ChevronDown className={`w-4 h-4 transition-transform ${isTodoExpanded ? '' : 'rotate-180'}`} />
                </button>
                <button className="text-blue-600 hover:text-blue-700 transition-colors">
                  <PlusCircle className="w-5 h-5" />
                </button>
              </div>
            </div>
            
            <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isTodoExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'}`}>
              <div className="space-y-3">
                <div className="bg-blue-50 dark:bg-blue-500/5 rounded-xl p-3 border border-blue-100 dark:border-blue-500/10 space-y-2">
                  <textarea 
                    value={newTodo}
                    onChange={(e) => setNewTodo(e.target.value)}
                    className="w-full bg-transparent border-none p-0 text-sm text-slate-900 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:ring-0 resize-none h-12 outline-none" 
                    placeholder="输入新的待办事项..."
                  ></textarea>
                  <div className="flex items-center justify-between pt-2 border-t border-blue-200 dark:border-blue-500/5">
                    <div className="flex gap-2">
                      <span className="w-5 h-5 rounded-full bg-blue-100 dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 flex items-center justify-center text-[10px] font-bold">张</span>
                      <span className="w-5 h-5 rounded-full bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 flex items-center justify-center text-[10px] font-bold">李</span>
                    </div>
                    <button onClick={handleAddTodo} className="px-3 py-1 bg-blue-600 rounded-lg text-xs font-bold hover:bg-blue-700 transition-colors text-white">发布到全体</button>
                  </div>
                </div>
                
                <div className="space-y-2">
                  {todos.map((todo) => (
                    <div key={todo.id} className="flex items-center gap-3 p-2 hover:bg-slate-100 dark:hover:bg-white/5 rounded-lg transition-colors group">
                      <GripVertical className="text-slate-400 dark:text-slate-500 w-4 h-4 cursor-grab" />
                      <input 
                        type="checkbox" 
                        checked={todo.checked} 
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setTodos(todos.map(t => t.id === todo.id ? { ...t, checked, status: checked ? '已完成' : '进行中' } : t));
                          handleUpdateTodo(todo.id, { status: checked ? '已完成' : '进行中' });
                        }} 
                        className="rounded border-blue-500/40 text-blue-600 focus:ring-blue-600 w-4 h-4" 
                      />
                      <input 
                        type="text" 
                        value={todo.text} 
                        onChange={(e) => {
                          setTodos(todos.map(t => t.id === todo.id ? { ...t, text: e.target.value } : t));
                        }}
                        onBlur={(e) => {
                          handleUpdateTodo(todo.id, { task: e.target.value });
                        }}
                        className="bg-transparent border-none p-0 text-sm text-slate-700 dark:text-slate-300 focus:ring-0 flex-1 outline-none" 
                      />
                      <button 
                        onClick={() => handleDeleteTodo(todo.id)}
                        className="opacity-0 group-hover:opacity-100 text-slate-400 dark:text-slate-500 hover:text-red-500 dark:hover:text-red-400"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Vote className="text-blue-600 w-5 h-5" />
                <h3 className="text-sm font-bold text-slate-900 dark:text-slate-200">投票管理</h3>
              </div>
              <button onClick={() => setIsVoteExpanded(!isVoteExpanded)} className="text-slate-400 hover:text-slate-600">
                <ChevronDown className={`w-4 h-4 transition-transform ${isVoteExpanded ? '' : 'rotate-180'}`} />
              </button>
            </div>
            
            <div className={`overflow-hidden transition-all duration-300 ease-in-out ${isVoteExpanded ? 'max-h-[1000px] opacity-100' : 'max-h-0 opacity-0'}`}>
              <div className="bg-slate-50 dark:bg-slate-800/50 rounded-xl p-4 border border-slate-200 dark:border-slate-700 space-y-4">
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold text-slate-500 uppercase">投票标题</label>
                  <input 
                    type="text" 
                    value={voteTitle}
                    onChange={(e) => setVoteTitle(e.target.value)}
                    placeholder="例如：确认投放平台首选" 
                    className="w-full bg-white dark:bg-slate-950/50 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-900 dark:text-slate-200 focus:border-blue-600 focus:ring-1 focus:ring-blue-600 outline-none" 
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-[10px] font-bold text-slate-500 uppercase">投票截止时间</label>
                  <select 
                    value={voteDuration}
                    onChange={(e) => setVoteDuration(e.target.value)}
                    className="w-full bg-white dark:bg-slate-950/50 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-900 dark:text-slate-200 focus:border-blue-600 focus:ring-1 focus:ring-blue-600 outline-none" 
                  >
                    <option value="1">1分钟</option>
                    <option value="2">2分钟</option>
                    <option value="3">3分钟</option>
                    <option value="5">5分钟</option>
                  </select>
                </div>
                <div className="space-y-3">
                  <label className="text-[10px] font-bold text-slate-500 uppercase">选项</label>
                  {voteOptions.map((opt, index) => (
                    <div key={opt.id} className="flex items-center gap-2">
                      <input 
                        type="text" 
                        value={opt.text}
                        onChange={(e) => handleVoteOptionChange(opt.id, e.target.value)}
                        placeholder={`选项${index + 1}`} 
                        className="flex-1 bg-white dark:bg-slate-950/50 border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-900 dark:text-slate-200 focus:border-blue-600 focus:ring-1 focus:ring-blue-600 outline-none" 
                      />
                      {voteOptions.length > 2 && (
                        <button 
                          onClick={() => handleRemoveVoteOption(opt.id)}
                          className="text-slate-400 hover:text-red-500 transition-colors"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  ))}
                  <button 
                    onClick={handleAddVoteOption}
                    className="w-full py-2 border border-dashed border-slate-300 dark:border-slate-600 rounded-lg text-xs text-slate-500 hover:border-blue-600 hover:text-blue-600 transition-colors flex items-center justify-center gap-1"
                  >
                    <Plus className="w-4 h-4" /> 添加选项
                  </button>
                </div>
                <button 
                  onClick={handleStartVote}
                  className="w-full py-2.5 bg-blue-600 hover:bg-blue-700 rounded-xl text-sm font-bold text-white transition-all shadow-lg shadow-blue-600/20"
                >
                  发起投票
                </button>
              </div>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-gradient-to-br from-blue-50 to-transparent dark:from-blue-600/20 dark:to-transparent border border-blue-100 dark:border-blue-500/20 mb-4">
            <p className="text-xs font-bold text-blue-600 mb-2 flex items-center gap-1">
              <Sparkles className="w-3 h-3" /> 会议助手
            </p>
            <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-400">
              王强正在发言，是否需要为其创建一个关于“KPI 审核”的专题投票？
            </p>
            <button className="mt-3 text-xs font-bold text-blue-600 hover:underline">点击创建</button>
          </div>
        </aside>
      </main>

      {/* Footer Controls */}
      {selectedVoteResult && (
        <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 rounded-2xl p-8 w-full max-w-md shadow-2xl border border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600">
                  <BarChart className="w-6 h-6" />
                </div>
                <h3 className="text-xl font-bold text-slate-900 dark:text-slate-100">{selectedVoteResult.title}</h3>
              </div>
              <button onClick={() => setSelectedVoteResult(null)} className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-lg transition-colors">
                <X className="w-5 h-5 text-slate-400" />
              </button>
            </div>
            <div className="space-y-4">
              {selectedVoteResult.options.map((opt) => {
                const totalVotes = selectedVoteResult.options.reduce((sum, o) => sum + o.votes, 0);
                const percentage = totalVotes > 0 ? Math.round((opt.votes / totalVotes) * 100) : 0;
                return (
                  <div key={opt.id} className="space-y-2">
                    <div className="flex justify-between text-sm font-bold">
                      <span className="text-slate-700 dark:text-slate-300">{opt.text}</span>
                      <span className="text-blue-600">{opt.votes} 票 ({percentage}%)</span>
                    </div>
                    <div className="h-2.5 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-blue-600 transition-all duration-1000" 
                        style={{ width: `${percentage}%` }}
                      ></div>
                    </div>
                  </div>
                );
              })}
            </div>
            <div className="mt-8 flex justify-center">
              <button 
                onClick={() => setSelectedVoteResult(null)}
                className="px-8 py-3 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-xl font-bold text-slate-700 dark:text-slate-200 transition-all"
              >
                关闭详情
              </button>
            </div>
          </div>
        </div>
      )}
      {/* Summary Modal */}
      {showSummaryModal && (
        <div className="fixed inset-0 z-[100] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl p-8 w-full max-w-lg shadow-2xl border border-slate-700 text-slate-100">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-bold">会议总结</h3>
              <button onClick={() => setShowSummaryModal(false)} className="text-slate-400 hover:text-slate-200">
                <X className="w-5 h-5" />
              </button>
            </div>
            {!summaryData && !isGeneratingSummary && (
              <button 
                onClick={handleEndMeeting}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 rounded-xl font-bold text-white transition-all"
              >
                生成总结
              </button>
            )}
            {isGeneratingSummary && (
              <div className="flex flex-col items-center gap-4 py-8">
                <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                <p className="text-slate-400">正在生成总结并保存至知识库...</p>
              </div>
            )}
            {summaryData && (
              <div className="space-y-6">
                <div>
                  <h4 className="font-bold text-blue-400 mb-2">决策</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-slate-300">
                    {summaryData.decisions.map((d: any, i: number) => <li key={i}>{typeof d === 'string' ? d : JSON.stringify(d)}</li>)}
                  </ul>
                </div>
                <div>
                  <h4 className="font-bold text-emerald-400 mb-2">行动项</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-slate-300">
                    {summaryData.actions.map((a: any, i: number) => (
                      <li key={i}>
                        {typeof a === 'string' ? a : `${a.task} (负责人: ${a.assignee || '待分配'}, 截止: ${a.dueDate || '无'})`}
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="flex gap-3 pt-4">
                  <button 
                    onClick={() => navigator.clipboard.writeText(`决策:\n${summaryData.decisions.join('\n')}\n\n行动项:\n${JSON.stringify(summaryData.actions)}`)}
                    className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl font-bold text-sm transition-all"
                  >
                    <Clipboard className="w-4 h-4" /> 复制到剪贴板
                  </button>
                  <button 
                    onClick={confirmEndMeeting}
                    className="flex-1 py-2.5 bg-red-600 hover:bg-red-700 rounded-xl font-bold text-sm transition-all"
                  >
                    确认结束会议
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      <footer className="bg-white dark:bg-slate-900 border-t border-blue-500/20 px-8 py-4 shrink-0">
        <div className="max-w-screen-xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-6">
            <button onClick={toggleMic} className={`flex flex-col items-center gap-1 group ${isMicOn ? 'text-blue-600' : 'text-slate-500 dark:text-slate-400 hover:text-blue-600'}`}>
              {isMicOn ? <Mic className="w-6 h-6" /> : <MicOff className="w-6 h-6" />}
              <span className="text-[10px] font-medium">麦克风</span>
            </button>
            <button onClick={toggleCamera} className={`flex flex-col items-center gap-1 group ${isCameraOn ? 'text-blue-600' : 'text-slate-500 dark:text-slate-400 hover:text-blue-600'}`}>
              {isCameraOn ? <Video className="w-6 h-6" /> : <VideoOff className="w-6 h-6" />}
              <span className="text-[10px] font-medium">摄像头</span>
            </button>
          </div>
          
          <div className="flex items-center gap-4">
            <input 
              type="text" 
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              placeholder="输入文本，发送消息至聊天界面..."
              className="w-64 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
            />
            <button 
              onClick={handleSendMessage}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-xl text-sm font-bold transition-colors"
            >
              发送
            </button>
            <button 
              onClick={() => setIsPaused(!isPaused)}
              className={`flex items-center gap-2 px-6 py-2.5 rounded-xl transition-all border font-bold text-sm ${
                isPaused 
                  ? 'bg-amber-100 border-amber-200 text-amber-700 dark:bg-amber-900/30 dark:border-amber-800 dark:text-amber-400' 
                  : 'bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-100'
              }`}
            >
              {isPaused ? <PlayCircle className="w-5 h-5" /> : <PauseCircle className="w-5 h-5" />}
              <span>{isPaused ? '继续录制' : '暂停录制'}</span>
            </button>
            
            <input 
              type="file" 
              ref={fileInputRef} 
              className="hidden" 
              onChange={handleFileChange} 
            />
            <button 
              onClick={handleUploadClick}
              className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 transition-all border border-blue-500/50 font-bold text-sm text-white shadow-lg shadow-blue-600/20"
            >
              <UploadCloud className="w-5 h-5" />
              <span>上传文件</span>
            </button>
            
            <button 
              onClick={handleEndMeeting}
              className="flex items-center gap-2 px-8 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 transition-all font-bold text-sm text-white shadow-lg shadow-red-600/20"
            >
              <PhoneOff className="w-5 h-5" />
              <span>结束会议</span>
            </button>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 bg-slate-50 dark:bg-slate-800/50 px-3 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700">
              <span className="text-xs font-medium text-slate-600 dark:text-slate-400">高亮发言人</span>
              <button className="relative inline-flex h-5 w-10 shrink-0 cursor-pointer items-center rounded-full bg-blue-600 transition-colors focus:outline-none">
                <span className="translate-x-5 pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out"></span>
              </button>
            </div>
            <button className="w-10 h-10 flex items-center justify-center rounded-xl bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:border-blue-500/50 transition-colors">
              <Settings className="text-slate-600 dark:text-slate-300 w-5 h-5" />
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}


