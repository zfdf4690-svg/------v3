import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DragDropContext, Droppable, Draggable, DropResult } from '@hello-pangea/dnd';
import { useMeetings } from '../contexts/MeetingContext';
import { NotificationModal } from '../components/NotificationModal';
import { 
  List, 
  Plus, 
  GripVertical, 
  MinusCircle, 
  Paperclip, 
  CloudUpload, 
  FileText, 
  Trash2, 
  Calendar, 
  UserPlus, 
  X, 
  Contact, 
  Send 
} from 'lucide-react';

interface AgendaItem {
  id: string;
  title: string;
  objective: string;
  duration: string;
}

export function MeetingPlanner() {
  const navigate = useNavigate();
  const { addMeeting } = useMeetings();
  const [title, setTitle] = useState('');
  const [agendaItems, setAgendaItems] = useState<AgendaItem[]>([
    { id: '1', title: '开场致辞与议程介绍', objective: '对齐季度目标', duration: '10min' },
    { id: '2', title: '核心功能讨论与反馈', objective: '确定 V2.0 路线图', duration: '30min' },
    { id: '3', title: '总结与行动项分配', objective: '责任到人', duration: '5min' },
  ]);
  const [newAgendaTitle, setNewAgendaTitle] = useState('');
  const [newAgendaObjective, setNewAgendaObjective] = useState('');
  const [newAgendaDuration, setNewAgendaDuration] = useState('10min');
  const [dateTime, setDateTime] = useState('2026-03-10T14:00');
  const [participants, setParticipants] = useState([
    { id: '1', name: 'Alice (主持人)', role: 'Organizer', avatar: 'Alice' },
    { id: '2', name: 'Bob Chen', role: '', avatar: 'Bob' },
  ]);
  const [newParticipant, setNewParticipant] = useState('');
  const [attachments, setAttachments] = useState<{id: string, name: string, size: string}[]>([]);
  const [isNotificationModalOpen, setIsNotificationModalOpen] = useState(false);

  const addAgendaItem = () => {
    if (!newAgendaTitle) return;
    const newItem: AgendaItem = {
      id: Date.now().toString(),
      title: newAgendaTitle,
      objective: newAgendaObjective,
      duration: newAgendaDuration,
    };
    setAgendaItems([...agendaItems, newItem]);
    setNewAgendaTitle('');
    setNewAgendaObjective('');
  };

  const removeAgendaItem = (id: string) => {
    setAgendaItems(agendaItems.filter(item => item.id !== id));
  };

  const onDragEnd = (result: DropResult) => {
    if (!result.destination) return;
    const items = Array.from(agendaItems);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    setAgendaItems(items);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setAttachments([...attachments, { id: Date.now().toString(), name: file.name, size: (file.size / 1024 / 1024).toFixed(2) + 'MB' }]);
    }
  };

  const removeAttachment = (id: string) => {
    setAttachments(attachments.filter(a => a.id !== id));
  };

  const addParticipant = () => {
    if (!newParticipant) return;
    setParticipants([...participants, { id: Date.now().toString(), name: newParticipant, role: '', avatar: newParticipant.charAt(0) }]);
    setNewParticipant('');
  };

  const removeParticipant = (id: string) => {
    setParticipants(participants.filter(p => p.id !== id));
  };

  const handlePublish = () => {
    addMeeting({
      id: Date.now().toString(),
      title: title || '未命名会议',
      startTime: '14:00 - 15:00', // Mock time
      status: 'pending',
      isHost: true,
      participants: participants.length,
      participantNames: participants.map(p => p.name),
      summary: "会议尚未开始，暂无总结。",
      decisions: [],
      takeaways: [],
      actionItems: [],
      votes: []
    });
    navigate('/meetings');
  };

  return (
    <div className="mx-auto max-w-4xl space-y-8 p-8">
      {/* Title */}
      <section className="space-y-4">
        <div className="flex items-end justify-between">
          <div className="flex-1 max-w-2xl">
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">会议标题</label>
            <input 
              type="text" 
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="如：Q1 产品回顾会议" 
              className="w-full rounded-xl border-slate-200 bg-white p-5 text-2xl font-bold placeholder:text-slate-300 focus:border-blue-600 focus:ring-0 dark:bg-slate-900 dark:border-slate-800 shadow-sm outline-none" 
            />
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-8">
          {/* Agenda */}
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-800">
            <div className="mb-6 flex flex-col gap-2">
              <div className="flex gap-2">
                <input value={newAgendaTitle} onChange={(e) => setNewAgendaTitle(e.target.value)} placeholder="议程标题" className="flex-1 rounded-lg border border-slate-200 p-2 text-sm dark:bg-slate-800 dark:border-slate-700" />
                <input value={newAgendaObjective} onChange={(e) => setNewAgendaObjective(e.target.value)} placeholder="核心目标" className="flex-1 rounded-lg border border-slate-200 p-2 text-sm dark:bg-slate-800 dark:border-slate-700" />
                <select value={newAgendaDuration} onChange={(e) => setNewAgendaDuration(e.target.value)} className="rounded-lg border border-slate-200 p-2 text-sm dark:bg-slate-800 dark:border-slate-700">
                  <option>5min</option><option>10min</option><option>15min</option><option>30min</option>
                </select>
                <button onClick={addAgendaItem} className="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-white hover:bg-blue-700 transition-all">
                  <Plus className="w-5 h-5" />
                </button>
              </div>
            </div>
            
            <DragDropContext onDragEnd={onDragEnd}>
              <Droppable droppableId="agenda">
                {(provided) => (
                  <div {...provided.droppableProps} ref={provided.innerRef} className="space-y-3">
                    {agendaItems.map((item, index) => (
                      // @ts-expect-error - key is a valid react prop
                      <Draggable key={item.id} draggableId={item.id} index={index}>
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            className={`group flex items-center gap-4 rounded-lg border p-4 transition-all ${
                              snapshot.isDragging 
                                ? 'border-blue-600/50 bg-blue-600/5 shadow-md' 
                                : 'border-slate-100 bg-slate-50/50 hover:border-blue-600/30 hover:bg-white dark:bg-slate-800 dark:border-slate-700'
                            }`}
                          >
                            <div {...provided.dragHandleProps} className="cursor-grab text-slate-300 group-hover:text-slate-400">
                              <GripVertical className="w-5 h-5" />
                            </div>
                            <div className="flex-1">
                              <p className="font-semibold">{item.title}</p>
                              <p className="text-xs text-slate-400">核心目标：{item.objective}</p>
                            </div>
                            <span className="rounded-md bg-slate-200 px-2 py-1 text-xs font-bold text-slate-600 dark:bg-slate-700 dark:text-slate-300">
                              {item.duration}
                            </span>
                            <button onClick={() => removeAgendaItem(item.id)} className="text-slate-400 hover:text-red-500 transition-opacity">
                              <MinusCircle className="w-5 h-5" />
                            </button>
                          </div>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </div>
                )}
              </Droppable>
            </DragDropContext>
          </section>

          {/* Attachments */}
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-800">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-bold">
              <Paperclip className="text-blue-600 w-5 h-5" />
              附件资料
            </h3>
            <div className="rounded-xl border-2 border-dashed border-slate-200 bg-slate-50 p-8 text-center transition-all hover:border-blue-600 hover:bg-blue-600/5 dark:bg-slate-800 dark:border-slate-700 cursor-pointer relative">
              <input type="file" onChange={handleFileUpload} className="absolute inset-0 opacity-0 cursor-pointer" />
              <CloudUpload className="w-10 h-10 text-slate-300 mb-2 mx-auto" />
              <p className="text-sm font-medium text-slate-600 dark:text-slate-300">点击或拖拽文件到此处上传</p>
              <p className="mt-1 text-xs text-slate-400">支持 PDF, DOCX, PPTX, Figma 链接 (最大 50MB)</p>
            </div>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              {attachments.map(file => (
                <div key={file.id} className="flex items-center gap-3 rounded-lg border border-slate-100 p-3 hover:shadow-sm dark:border-slate-800">
                  <FileText className="text-red-500 w-5 h-5" />
                  <div className="flex-1 overflow-hidden">
                    <p className="truncate text-sm font-medium">{file.name}</p>
                    <p className="text-[10px] text-slate-400">{file.size}</p>
                  </div>
                  <button onClick={() => removeAttachment(file.id)} className="text-slate-300 hover:text-red-500 transition-colors">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </section>
        </div>

        <div className="space-y-8">
          {/* Time Settings */}
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-800">
            <h3 className="mb-4 text-sm font-bold uppercase tracking-wider text-slate-400">时间设置</h3>
            <div className="space-y-4">
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-500">会议开始日期与时间</label>
                <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 dark:bg-slate-800 dark:border-slate-700">
                  <Calendar className="text-slate-400 w-4 h-4" />
                  <input type="datetime-local" value={dateTime} onChange={(e) => setDateTime(e.target.value)} className="w-full border-none bg-transparent p-0 text-sm focus:ring-0 outline-none" />
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-xs font-medium text-slate-500">时区</label>
                <select className="w-full rounded-lg border-slate-200 bg-slate-50 py-2 text-sm focus:border-blue-600 focus:ring-0 dark:bg-slate-800 dark:border-slate-700 outline-none">
                  <option>(GMT+08:00) 北京, 上海</option>
                  <option>(GMT+00:00) 伦敦</option>
                  <option>(GMT-08:00) 洛杉矶</option>
                </select>
              </div>
              <label className="flex items-center gap-2 cursor-pointer pt-2">
                <input type="checkbox" defaultChecked className="rounded border-slate-300 text-blue-600 focus:ring-blue-600 w-4 h-4" />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">立即向全员发送通知</span>
              </label>
            </div>
          </section>

          {/* Participants */}
          <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm dark:bg-slate-900 dark:border-slate-800">
            <h3 className="mb-4 text-sm font-bold uppercase tracking-wider text-slate-400">参会人员</h3>
            <div className="relative mb-4">
              <input type="text" value={newParticipant} onChange={(e) => setNewParticipant(e.target.value)} placeholder="输入姓名或邮箱..." className="w-full rounded-lg border-slate-200 bg-slate-50 py-2 pl-3 pr-10 text-sm focus:border-blue-600 focus:ring-0 dark:bg-slate-800 dark:border-slate-700 outline-none" />
              <button onClick={addParticipant} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-blue-600">
                <UserPlus className="w-5 h-5" />
              </button>
            </div>
            <div className="space-y-3 mb-4">
              {participants.map(p => (
                    <div key={p.id} className="flex items-center justify-between">
                      <div className="flex items-center gap-2 cursor-pointer" onClick={addParticipant}>
                        <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${p.avatar}`} alt={p.name} className="h-7 w-7 rounded-full bg-slate-100" />
                        <span className="text-xs font-medium">{p.name}</span>
                      </div>
                      {p.role !== 'Organizer' && (
                        <button onClick={() => removeParticipant(p.id)} className="text-slate-300 hover:text-red-500">
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>
              ))}
            </div>
            <button className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 py-2 text-xs font-bold text-slate-600 transition-colors hover:bg-slate-50 dark:border-slate-700 dark:text-slate-400 dark:hover:bg-slate-800">
              <Contact className="w-4 h-4" />
              从通讯录导入
            </button>
          </section>
        </div>
      </div>

      {/* Action Button */}
      <div className="sticky bottom-8 mt-12 flex justify-center">
        <div className="flex flex-col items-center gap-4">
          <button 
            onClick={() => setIsNotificationModalOpen(true)}
            className="group relative flex min-w-[320px] items-center justify-center gap-3 overflow-hidden rounded-full bg-blue-600 px-10 py-4 text-lg font-bold text-white shadow-xl shadow-blue-600/30 transition-all hover:scale-105 active:scale-95"
          >
            <Send className="w-5 h-5 group-hover:animate-pulse" />
            <span>发布会议并发送通知</span>
          </button>
          <NotificationModal 
            isOpen={isNotificationModalOpen}
            onClose={() => setIsNotificationModalOpen(false)}
            onConfirm={handlePublish}
            title={title}
            time={dateTime}
            agendas={agendaItems.map(item => ({ title: item.title, duration: item.duration }))}
          />
          <p className="text-xs font-medium text-slate-400">上次草稿自动保存于 14:24:05</p>
        </div>
      </div>
    </div>
  );
}
