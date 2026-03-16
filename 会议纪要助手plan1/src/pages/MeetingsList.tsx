import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMeetings } from '../contexts/MeetingContext';
import { Video, Clock, Play, Calendar, Users } from 'lucide-react';

export function MeetingsList() {
  const { meetings } = useMeetings();
  const navigate = useNavigate();

  return (
    <div className="p-8 max-w-5xl mx-auto w-full">
      <div className="mb-8">
        <h2 className="text-2xl font-bold">我的会议</h2>
        <p className="text-slate-500 mt-1">查看和管理与您相关的会议。</p>
      </div>

      <div className="space-y-4">
        {meetings.map(meeting => (
          <div key={meeting.id} className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row md:items-center justify-between shadow-sm hover:shadow-md transition-all gap-4">
            <div className="flex items-center gap-6">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${
                meeting.status === 'pending' ? 'bg-blue-100 text-blue-600 dark:bg-blue-900/30' : 'bg-slate-100 text-slate-500 dark:bg-slate-800'
              }`}>
                <Calendar className="w-6 h-6" />
              </div>
              <div>
                <div className="flex items-center gap-3 mb-1 flex-wrap">
                  <h3 className="text-lg font-bold">{meeting.title}</h3>
                  {meeting.status === 'pending' && (
                    <span className="px-2 py-0.5 bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 text-xs font-bold rounded">待开始</span>
                  )}
                  {meeting.status === 'not_started' && (
                    <span className="px-2 py-0.5 bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400 text-xs font-bold rounded">未开始</span>
                  )}
                  {meeting.isHost && (
                    <span className="px-2 py-0.5 bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400 text-xs font-bold rounded">主持人</span>
                  )}
                </div>
                <div className="flex items-center gap-4 text-sm text-slate-500">
                  <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> {meeting.startTime}</span>
                  <span className="flex items-center gap-1"><Users className="w-4 h-4" /> {meeting.participants} 人参会</span>
                </div>
              </div>
            </div>
            
            <div className="flex items-center gap-3 shrink-0">
              {meeting.isHost ? (
                <button 
                  onClick={() => navigate(`/meetings/${meeting.id}/active`)}
                  className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-5 py-2.5 rounded-xl font-bold text-sm transition-all shadow-md shadow-blue-600/20"
                >
                  <Play className="w-4 h-4 fill-current" />
                  开播
                </button>
              ) : (
                <button 
                  onClick={() => navigate(`/meetings/${meeting.id}/active`)}
                  className="flex items-center gap-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 px-5 py-2.5 rounded-xl font-bold text-sm transition-all"
                >
                  <Video className="w-4 h-4" />
                  进入会议
                </button>
              )}
            </div>
          </div>
        ))}
        
        {meetings.length === 0 && (
          <div className="text-center py-20 text-slate-500">
            暂无相关会议
          </div>
        )}
      </div>
    </div>
  );
}
