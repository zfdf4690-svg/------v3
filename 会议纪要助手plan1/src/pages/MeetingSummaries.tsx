import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMeetings } from '../contexts/MeetingContext';
import { Calendar, Clock, Users, ChevronRight } from 'lucide-react';

export function MeetingSummaries() {
  const { meetings } = useMeetings();
  const navigate = useNavigate();
  
  const finishedMeetings = (meetings || []).filter(m => m.status === 'finished');

  return (
    <div className="p-8 max-w-5xl mx-auto w-full">
      <div className="mb-8">
        <h2 className="text-2xl font-bold">会议总结</h2>
        <p className="text-slate-500 mt-1">点击会议条例查看详细总结。</p>
      </div>

      <div className="space-y-4">
        {finishedMeetings.map(meeting => (
          <div 
            key={meeting.id} 
            onClick={() => navigate(`/meetings/${meeting.id}`)}
            className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-6 rounded-2xl flex items-center justify-between shadow-sm hover:shadow-md transition-all cursor-pointer group"
          >
            <div className="flex items-center gap-6">
              <div className="w-12 h-12 rounded-xl flex items-center justify-center shrink-0 bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30">
                <Calendar className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-lg font-bold group-hover:text-blue-600 transition-colors">{meeting.title}</h3>
                <div className="flex items-center gap-4 text-sm text-slate-500 mt-1">
                  <span className="flex items-center gap-1"><Clock className="w-4 h-4" /> {meeting.startTime}</span>
                  <span className="flex items-center gap-1"><Users className="w-4 h-4" /> {meeting.participants} 人参会</span>
                </div>
              </div>
            </div>
            <ChevronRight className="text-slate-400 group-hover:text-blue-600 transition-colors" />
          </div>
        ))}
        
        {finishedMeetings.length === 0 && (
          <div className="text-center py-20 text-slate-500">
            暂无已完成的会议总结
          </div>
        )}
      </div>
    </div>
  );
}
