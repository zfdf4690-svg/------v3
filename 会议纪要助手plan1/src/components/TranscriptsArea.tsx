import React, { useState, useEffect, useRef } from 'react';

interface Bullet {
  id: string;
  text: string;
  top: number;
  color: string;
}

export function TranscriptsArea({ isMicOn, onTranscript }: { isMicOn: boolean, onTranscript: (text: string) => void }) {
  const [bullets, setBullets] = useState<Bullet[]>([]);
  const containerRef = useRef<HTMLDivElement>(null);
  
  // 用于界面显示“正在说话...”的实时中间气泡文本
  const [interimUI, setInterimUI] = useState('');

  // 借助 ref 防止闭包里取到旧数据
  const finalBufferRef = useRef('');
  const currentInterimRef = useRef('');
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // 真正的输出函数
  const flushText = () => {
    // 汇总完整的句子：已确认的 Final 加上最后的 Interim
    const textToOutput = (finalBufferRef.current + currentInterimRef.current).trim();
    if (textToOutput) {
      console.log('静音3秒，输出句子：' + textToOutput);
      onTranscript(textToOutput);
      addBullet(textToOutput);
    }
    // 输出后清空积累的文本
    finalBufferRef.current = '';
    currentInterimRef.current = '';
    setInterimUI('');
  };

  useEffect(() => {
    if (!isMicOn) return;

    let recognition: any;
    let errorCount = 0;
    const maxErrors = 5;

    const startRecognition = () => {
      if (errorCount >= maxErrors) {
        alert('语音识别异常，请尝试刷新页面或检查麦克风权限');
        return;
      }

      try {
        recognition = new (window.SpeechRecognition || (window as any).webkitSpeechRecognition)();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'zh-CN';

        recognition.onstart = () => {
          errorCount = 0; // 重置错误计数
        };

        recognition.onresult = (event: any) => {
          let newlyFinalized = '';
          let newlyInterim = '';

          for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
              newlyFinalized += event.results[i][0].transcript;
            } else {
              newlyInterim += event.results[i][0].transcript;
            }
          }

          // 如果有新的最终结果确认，追加到稳定 buffer 中
          if (newlyFinalized) {
            finalBufferRef.current += newlyFinalized;
          }
          // 当前尚未确认的句段更新 interim 缓存
          currentInterimRef.current = newlyInterim;

          // 更新要展示的总体实时文字 UI
          const currentTotalStr = finalBufferRef.current + currentInterimRef.current;
          setInterimUI(currentTotalStr);

          // 只要有结果更新，就重置3秒的定时器
          if (timerRef.current) {
            clearTimeout(timerRef.current);
          }
          
          if (currentTotalStr.trim()) {
            timerRef.current = setTimeout(() => {
              flushText();
            }, 3000);
          }
        };

        recognition.onerror = (event: any) => {
          console.error('Speech recognition error:', event.error);
          errorCount++;
        };

        recognition.onend = () => {
          // 只在用户依然开麦且没有频繁报错时，才自动重启
          if (isMicOn && errorCount < maxErrors) {
             console.log('识别暂停，自动重启');
             startRecognition();
          }
        };

        recognition.start();
      } catch (err) {
        console.error('Failed to start recognition:', err);
      }
    };

    startRecognition();

    return () => {
      // 卸载组件或关麦时，清理所有的定时器和监听事件
      if (timerRef.current) clearTimeout(timerRef.current);
      if (recognition) {
        recognition.onend = null; 
        recognition.onerror = null;
        recognition.stop();
      }
    };
  }, [isMicOn]); // 注意去掉了 onTranscript，以免其变动引起不断重载

  const addBullet = (text: string) => {
    const newBullet: Bullet = {
      id: Date.now().toString(),
      text,
      top: Math.random() * 80,
      color: `hsl(${Math.random() * 360}, 70%, 70%)`
    };
    setBullets(prev => [...prev, newBullet]);
    setTimeout(() => {
      setBullets(prev => prev.filter(b => b.id !== newBullet.id));
    }, 5000);
  };

  return (
    <div className="relative w-full h-64 bg-slate-950 border border-slate-800 rounded-xl overflow-hidden" ref={containerRef}>
      {bullets.map(bullet => (
        <div
          key={bullet.id}
          className="absolute whitespace-nowrap font-bold text-lg animate-bullet"
          style={{
            top: `${bullet.top}%`,
            color: bullet.color,
            animation: 'bullet-move 5s linear forwards'
          }}
        >
          {bullet.text}
        </div>
      ))}
      
      {/* 实时显示的中间结果气泡 */}
      {interimUI && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 max-w-[90%] bg-blue-600/90 backdrop-blur-sm text-white px-4 py-2 rounded-2xl shadow-lg border border-blue-400/50 flex items-center gap-3 z-10">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 animate-pulse shrink-0 shadow-sm shadow-red-500/50"></span>
          <span className="text-sm font-medium line-clamp-2">{interimUI}</span>
        </div>
      )}

      <style>{`
        @keyframes bullet-move {
          from { left: 100%; }
          to { left: -100%; }
        }
        .animate-bullet { animation: bullet-move 5s linear forwards; }
      `}</style>
    </div>
  );
}
