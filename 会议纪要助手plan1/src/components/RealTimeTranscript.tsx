import React, { useState, useEffect, useRef } from 'react';
import { Loader2 } from "lucide-react";

interface Transcript {
  id: string;
  speaker: string;
  time: string;
  text: string;
}

const fakeTranscripts: Transcript[] = [
  { id: '1', speaker: "李芳", time: "10:45:23", text: "是的，张伟。我已经整理好了..." },
  { id: '2', speaker: "张伟", time: "10:45:45", text: "那我们下周三定吧。" },
  { id: '3', speaker: "王强", time: "10:46:10", text: "关于供应商的资质审核，我建议更严格。" }
];

export function RealTimeTranscript() {
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [isTranscribing, setIsTranscribing] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < fakeTranscripts.length) {
        setTranscripts(prev => [...prev, fakeTranscripts[index]]);
        index++;
      } else {
        setIsTranscribing(false);
        clearInterval(interval);
      }
    }, 4000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [transcripts]);

  return (
    <div className="flex flex-col h-full bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
      <div className="p-4 border-b border-slate-800 flex justify-between items-center">
        <h3 className="text-sm font-bold text-slate-200">实时AI转写</h3>
        {isTranscribing && (
          <div className="flex items-center gap-2 text-xs text-blue-400">
            <Loader2 className="w-3 h-3 animate-spin" />
            转写中...
          </div>
        )}
      </div>
      <div className="flex-1 p-4 overflow-y-auto" ref={scrollRef}>
        <div className="space-y-4">
          {transcripts.map((t) => (
            <div key={t.id} className="bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-bold text-blue-300">{t.speaker}</span>
                <span className="text-[10px] text-slate-500">{t.time}</span>
              </div>
              <p className="text-sm text-slate-300">{t.text}</p>
            </div>
          ))}
          {!isTranscribing && transcripts.length === 0 && (
            <p className="text-center text-slate-500 text-sm py-10">暂无转写内容</p>
          )}
        </div>
      </div>
    </div>
  );
}
