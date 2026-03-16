import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { useState } from "react"
import { toast } from "sonner"
import { useNavigate } from "react-router-dom"

interface AgendaItem {
  title: string;
  duration: string;
}

interface MeetingNotificationDialogProps {
  title: string;
  time: string;
  agendas: AgendaItem[];
}

export function MeetingNotificationDialog({ title, time, agendas }: MeetingNotificationDialogProps) {
  const [selectedPlatforms, setSelectedPlatforms] = useState<string[]>(["dingtalk"]);
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();

  const totalDuration = agendas.reduce((acc, curr) => {
    const duration = parseInt(curr.duration.replace('min', ''));
    return acc + (isNaN(duration) ? 0 : duration);
  }, 0);

  const handleSend = () => {
    if (selectedPlatforms.length === 0) {
        toast.error("至少选一个");
        return;
    }
    console.log("发送到 " + selectedPlatforms.join(", "));
    
    if (selectedPlatforms.includes("wechat")) {
        navigator.clipboard.writeText("https://example.com/meeting/123");
    }
    
    setOpen(false);
    toast.success("通知已发送！");
    navigate("/meeting/123");
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="rounded-full px-10 py-4 text-lg font-bold">发送会议通知</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[400px] dark">
        <DialogHeader>
          <DialogTitle>发送会议通知</DialogTitle>
          <DialogDescription>
            <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-800 rounded-lg text-sm text-slate-700 dark:text-slate-300">
                <p><strong>会议：</strong>{title || '未命名会议'}</p>
                <p><strong>时间：</strong>{time}</p>
                <p><strong>总时长：</strong>{totalDuration} 分钟</p>
            </div>
          </DialogDescription>
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
            <div className="flex items-center space-x-2">
                <Checkbox 
                    id="wechat" 
                    checked={selectedPlatforms.includes("wechat")}
                    onCheckedChange={(checked) => {
                        if (checked) setSelectedPlatforms([...selectedPlatforms, "wechat"]);
                        else setSelectedPlatforms(selectedPlatforms.filter(p => p !== "wechat"));
                    }}
                />
                <Label htmlFor="wechat">微信</Label>
            </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)}>取消</Button>
          <Button onClick={handleSend}>发送</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
