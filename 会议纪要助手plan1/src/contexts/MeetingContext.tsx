import React, { createContext, useContext, useState } from 'react';

export type MeetingStatus = 'pending' | 'not_started' | 'active' | 'finished';

export interface Decision {
  id: string;
  title: string;
  text: string;
}

export interface Takeaway {
  id: string;
  title: string;
  text: string;
}

export interface ActionItem {
  id: string;
  task: string;
  status: string;
  assignee: string;
  dueDate: string;
}

export interface VoteResult {
  id: string;
  title: string;
  options: { id: string, text: string, votes: number }[];
  totalVotes: number;
  winner: string;
}

export interface Meeting {
  id: string;
  title: string;
  startTime: string;
  status: MeetingStatus;
  isHost: boolean;
  participants: number;
  participantNames: string[];
  summary: string;
  decisions: Decision[];
  takeaways: Takeaway[];
  actionItems: ActionItem[];
  votes: VoteResult[];
}

export interface KnowledgeBaseEntry {
  id: string;
  meetingId: string;
  title: string;
  date: string;
  summary: boolean;
  audio: boolean;
  chat: boolean;
}

interface MeetingContextType {
  meetings: Meeting[];
  addMeeting: (meeting: Meeting) => void;
  endMeeting: (id: string, actionItems?: ActionItem[], votes?: VoteResult[]) => void;
  knowledgeBaseEntries: KnowledgeBaseEntry[];
  addKnowledgeBaseEntry: (entry: KnowledgeBaseEntry) => void;
}

const MeetingContext = createContext<MeetingContextType | undefined>(undefined);

export function MeetingProvider({ children }: { children: React.ReactNode }) {
  const [meetings, setMeetings] = useState<Meeting[]>([
    {
      id: '1',
      title: '产品迭代讨论会',
      startTime: '14:00 - 15:30',
      status: 'finished',
      isHost: true,
      participants: 6,
      participantNames: ['张经理', '李研发', '王设计', '赵产品', '其他1', '其他2'],
      summary: "本次会议主要讨论了产品迭代计划。团队确认了 V2.1 版本的发布时间，并针对仪表盘加载性能、移动端 UI 设计以及 AI 增强功能进行了深入探讨，达成了多项关键决策与待办事项。",
      decisions: [
        { id: '1', title: '通过：采用混合云部署架构', text: '针对大客户版本采用混合云部署，以满足数据合规性要求并确保性能。' },
        { id: '2', title: '通过：取消旧版 Webhook 维护', text: '从 V2.1 版本起全面停止对 v1.0 接口的技术支持，发布迁移指南。' },
      ],
      takeaways: [
        { id: '1', title: '发布计划', text: '确认了 V2.1 版本的发布日期定于 11月15日，所有核心功能需在 11月5日 前完成内测并输出报告。' },
        { id: '2', title: '性能优化', text: '研发团队将针对用户反馈的仪表盘加载速度缓慢问题，优先进行大规模接口并发性能优化与前端缓存策略调整。' },
      ],
      actionItems: [
        { id: '1', task: '整理 API 文档变更记录并分发', status: '进行中', assignee: '李研发', dueDate: '2023-10-27' },
        { id: '2', task: '移动端 UI 高保真稿最终交付', status: '未开始', assignee: '王设计', dueDate: '2023-11-01' },
      ],
      votes: [
        {
          id: 'v1',
          title: '确认投放平台首选',
          options: [
            { id: '1', text: '抖音', votes: 12 },
            { id: '2', text: '小红书', votes: 8 },
            { id: '3', text: 'B站', votes: 4 }
          ],
          totalVotes: 24,
          winner: '抖音'
        }
      ]
    },
    {
      id: '2',
      title: 'AI 特性用户调研访谈 - 第2轮',
      startTime: '16:30 - 17:30',
      status: 'not_started',
      isHost: false,
      participants: 2,
      participantNames: ['张经理', '李研发'],
      summary: "本次会议主要讨论了AI特性用户调研访谈。",
      decisions: [],
      takeaways: [],
      actionItems: [],
      votes: []
    }
  ]);
  const [knowledgeBaseEntries, setKnowledgeBaseEntries] = useState<KnowledgeBaseEntry[]>([]);

  const addMeeting = (meeting: Meeting) => {
    setMeetings(prev => [meeting, ...prev]);
  };

  const endMeeting = (id: string, actionItems?: ActionItem[], votes?: VoteResult[]) => {
    setMeetings(prev => prev.map(m => m.id === id ? { 
      ...m, 
      status: 'finished',
      actionItems: actionItems || m.actionItems,
      votes: votes || m.votes
    } : m));
  };

  const addKnowledgeBaseEntry = (entry: KnowledgeBaseEntry) => {
    setKnowledgeBaseEntries(prev => [entry, ...prev]);
  };

  return (
    <MeetingContext.Provider value={{ meetings, addMeeting, endMeeting, knowledgeBaseEntries, addKnowledgeBaseEntry }}>
      {children}
    </MeetingContext.Provider>
  );
}

export function useMeetings() {
  const context = useContext(MeetingContext);
  if (!context) throw new Error('useMeetings must be used within MeetingProvider');
  return context;
}
