import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { toast } from "sonner"

interface AgendaItem {
  title: string;
  duration: string;
}

interface NotificationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  time: string;
  agendas: AgendaItem[];
}

export function NotificationModal({ isOpen, onClose, onConfirm, title, time, agendas }: NotificationModalProps) {
  const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>(["dingtalk"]);

  const totalDuration = agendas.reduce((acc, curr) => {
    const duration = parseInt(curr.duration.replace('min', ''));
    return acc + (isNaN(duration) ? 0 : duration);
  }, 0);

  const handleSend = () => {
    if (selectedPlatforms.length === 0) {
      alert('至少选一个！');
      return;
    }
    
    console.log("选中的平台:", selectedPlatforms);
    
    // 模拟发送逻辑
    selectedPlatforms.forEach(platform => {
        const platformName = platform === 'dingtalk' ? '钉钉' : '飞书';
        console.log(`发送到${platformName}：会议通知已推送`);
    });
    
    toast("通知已发送！");
    onConfirm();
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[400px] dark text-slate-100">
        <DialogHeader>
          <DialogTitle>发送会议通知</DialogTitle>
          <div className="text-sm text-slate-400">
            <div className="mt-4 p-3 bg-slate-800 rounded-lg text-sm text-slate-300">
                <p><strong>会议：</strong>{title || '未命名会议'}</p>
                <p><strong>时间：</strong>{time}</p>
                <p><strong>总时长：</strong>{totalDuration} 分钟</p>
            </div>
          </div>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
            <div className="flex items-center space-x-2">
                <Checkbox 
                    id="dingtalk" 
                    checked={selectedPlatforms.includes("dingtalk")}
                    onCheckedChange={(checked) => {
                        if (checked) setSelectedPlatforms([...selectedPlatforms, "dingtalk"]);
                        else setSelectedPlatforms(selectedPlatforms.filter(p => p !== "dingtalk"));
                    }}
                />
                <Label htmlFor="dingtalk">钉钉群</Label>
            </div>
            <div className="flex items-center space-x-2">
                <Checkbox 
                    id="feishu" 
                    checked={selectedPlatforms.includes("feishu")}
                    onCheckedChange={(checked) => {
                        if (checked) setSelectedPlatforms([...selectedPlatforms, "feishu"]);
                        else setSelectedPlatforms(selectedPlatforms.filter(p => p !== "feishu"));
                    }}
                />
                <Label htmlFor="feishu">飞书群</Label>
            </div>
            <div className="flex items-center space-x-2 opacity-50 cursor-not-allowed">
                <Checkbox id="wechat" disabled />
                <Label htmlFor="wechat" className="cursor-not-allowed">微信 <span className="text-xs">(网页暂不支持直发，需小程序或企业微信)</span></Label>
            </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose}>取消</Button>
          <Button onClick={handleSend}>发送</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
