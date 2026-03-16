import React, { useState } from 'react';
import { Users, MessageSquare, Shield, Search, Send, Plus, UserPlus, Trash2, AlertTriangle, X, Paperclip } from 'lucide-react';

const initialGroups = [
  { id: '1', name: '产品研发组', members: 12, admin: '张经理', memberList: ['张经理', '李研发', '赵产品'] },
  { id: '2', name: 'UI 设计组', members: 5, admin: '王设计', memberList: ['王设计', 'UI设计师A'] },
  { id: '3', name: 'AI 算法组', members: 8, admin: '李研发', memberList: ['李研发', '算法工程师B'] },
  { id: '4', name: '电商', members: 1, admin: '我', memberList: ['我'] },
];

export function MyGroups() {
  const [groups, setGroups] = useState(initialGroups);
  const [selectedGroup, setSelectedGroup] = useState(groups[0]);
  const [messages, setMessages] = useState<Record<string, { sender: string, text: string }[]>>({
    '4': [{ sender: '系统', text: '增加文件上传功能，支持pdf word, Excel文档格式, 和图片格式。' }]
  });
  const [newMessage, setNewMessage] = useState('');
  const [showDissolveModal, setShowDissolveModal] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showInviteModal, setShowInviteModal] = useState(false);
  const [newGroupName, setNewGroupName] = useState('');
  const [mockFriends] = useState([
    { id: 'f1', name: '张三' },
    { id: 'f2', name: '李四' },
    { id: 'f3', name: '王五' },
  ]);

  const handleInvite = (friendName: string) => {
    const updatedGroups = groups.map(g => {
      if (g.id === selectedGroup.id) {
        return { ...g, members: g.members + 1, memberList: [...g.memberList, friendName] };
      }
      return g;
    });
    setGroups(updatedGroups);
    setSelectedGroup(updatedGroups.find(g => g.id === selectedGroup.id)!);
    setShowInviteModal(false);
  };

  const handleRemoveMember = (memberName: string) => {
    const updatedGroups = groups.map(g => {
      if (g.id === selectedGroup.id) {
        return { 
          ...g, 
          members: g.members - 1, 
          memberList: g.memberList.filter(m => m !== memberName) 
        };
      }
      return g;
    });
    setGroups(updatedGroups);
    setSelectedGroup(updatedGroups.find(g => g.id === selectedGroup.id)!);
  };

  const handleSendMessage = () => {
    if (!newMessage.trim()) return;
    setMessages(prev => ({
      ...prev,
      [selectedGroup.id]: [...(prev[selectedGroup.id] || []), { sender: '我', text: newMessage }]
    }));
    setNewMessage('');
  };

  const handleCreateGroup = () => {
    if (!newGroupName.trim()) return;
    const newGroup = {
      id: Date.now().toString(),
      name: newGroupName,
      members: 1,
      admin: '我',
      memberList: ['我']
    };
    setGroups([...groups, newGroup]);
    setSelectedGroup(newGroup);
    setNewGroupName('');
    setShowCreateModal(false);
  };

  const handleDissolveGroup = () => {
    const remainingGroups = groups.filter(g => g.id !== selectedGroup.id);
    setGroups(remainingGroups);
    setSelectedGroup(remainingGroups[0] || null);
    setShowDissolveModal(false);
  };

  return (
    <div className="flex h-full bg-white dark:bg-slate-900">
      {/* Group List */}
      <div className="w-80 border-r border-slate-200 dark:border-slate-800 flex flex-col">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <h2 className="text-xl font-black text-slate-900 dark:text-white">我的小组</h2>
          <button onClick={() => setShowCreateModal(true)} className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            <Plus className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-2">
          {groups.map(group => (
            <button
              key={group.id}
              onClick={() => setSelectedGroup(group)}
              className={`p-4 rounded-xl text-left transition-colors ${selectedGroup.id === group.id ? 'bg-blue-50 dark:bg-blue-900/20' : 'hover:bg-slate-50 dark:hover:bg-slate-800'}`}
            >
              <p className="font-bold text-slate-900 dark:text-white">{group.name}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{group.members} 名成员 · 管理员: {group.admin}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Chat Interface */}
      <div className="flex-1 flex flex-col">
        <div className="p-6 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <h3 className="text-lg font-black text-slate-900 dark:text-white">{selectedGroup?.name || '未选择小组'}</h3>
          {selectedGroup && (
            <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
              <Users className="w-4 h-4" />
              <span>{selectedGroup.members} 成员</span>
              <Shield className="w-4 h-4 ml-4" />
              <span>管理员: {selectedGroup.admin}</span>
            </div>
          )}
        </div>
        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-4">
          {(selectedGroup && messages[selectedGroup.id] ? messages[selectedGroup.id] : []).map((msg, i) => (
            <div key={i} className={`p-4 rounded-lg max-w-md ${msg.sender === '我' ? 'bg-blue-600 text-white self-end' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 self-start'}`}>
              <p className="text-sm">{msg.text}</p>
            </div>
          ))}
        </div>
        {selectedGroup && (
          <div className="p-6 border-t border-slate-200 dark:border-slate-800">
            <div className="relative flex items-center gap-2">
              <input type="file" id="file-upload" className="hidden" onChange={(e) => console.log(e.target.files)} />
              <label htmlFor="file-upload" className="p-2 text-slate-500 hover:text-blue-600 transition-colors cursor-pointer">
                <Paperclip className="w-5 h-5" />
              </label>
              <input 
                type="text" 
                value={newMessage}
                onChange={(e) => setNewMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="输入消息..." 
                className="flex-1 bg-slate-100 dark:bg-slate-800 rounded-lg pl-4 pr-12 py-3 text-sm outline-none" 
              />
              <button onClick={handleSendMessage} className="absolute right-2 top-1/2 -translate-y-1/2 p-2 bg-blue-600 text-white rounded-lg">
                <Send className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Member Display & Invite Panel */}
      <div className="w-72 border-l border-slate-200 dark:border-slate-800 p-6 flex flex-col gap-6">
        <h4 className="font-bold text-slate-900 dark:text-white">小组成员</h4>
        {selectedGroup && (
          <button onClick={() => setShowInviteModal(true)} className="flex items-center gap-2 w-full p-3 rounded-lg bg-blue-600 text-white text-sm font-bold justify-center hover:bg-blue-700 transition-colors">
            <UserPlus className="w-4 h-4" />
            邀请新成员
          </button>
        )}
        <div className="flex-1 overflow-y-auto space-y-4">
          {selectedGroup?.memberList.map((member, index) => (
            <div key={index} className="flex items-center gap-3 justify-between">
              <div className="flex items-center gap-3">
                <img src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${member}`} className="w-8 h-8 rounded-full bg-slate-100" />
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{member}</span>
                {member === selectedGroup.admin && <span className="text-[10px] bg-slate-200 dark:bg-slate-700 px-2 py-0.5 rounded-full">管理员</span>}
              </div>
              {member !== '我' && (
                <button onClick={() => handleRemoveMember(member)} className="text-slate-400 hover:text-red-500 transition-colors">
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}
        </div>
        {selectedGroup && (
          <button onClick={() => setShowDissolveModal(true)} className="flex items-center gap-2 w-full p-3 rounded-lg bg-red-50 text-red-600 text-sm font-bold justify-center hover:bg-red-100 transition-colors border border-red-200">
            <Trash2 className="w-4 h-4" />
            解散小组
          </button>
        )}
      </div>

      {/* Modals */}
      {showDissolveModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-slate-900 p-8 rounded-2xl shadow-xl max-w-sm w-full">
            <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-bold text-center mb-2">确定要解散小组吗？</h3>
            <p className="text-sm text-slate-500 text-center mb-6">此操作不可撤销，小组所有数据将丢失。</p>
            <div className="flex gap-4">
              <button onClick={() => setShowDissolveModal(false)} className="flex-1 p-3 rounded-lg border border-slate-200">取消</button>
              <button onClick={handleDissolveGroup} className="flex-1 p-3 rounded-lg bg-red-600 text-white">确定解散</button>
            </div>
          </div>
        </div>
      )}

      {showInviteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-slate-900 p-8 rounded-2xl shadow-xl max-w-sm w-full">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-bold">邀请好友</h3>
              <button onClick={() => setShowInviteModal(false)}><X className="w-5 h-5" /></button>
            </div>
            <div className="space-y-3">
              {mockFriends.map(friend => (
                <button key={friend.id} onClick={() => handleInvite(friend.name)} className="w-full p-3 rounded-lg border border-slate-200 hover:bg-blue-50 dark:hover:bg-blue-900/20 text-left">
                  {friend.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {showCreateModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-white dark:bg-slate-900 p-8 rounded-2xl shadow-xl max-w-sm w-full">
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-lg font-bold">创建新小组</h3>
              <button onClick={() => setShowCreateModal(false)}><X className="w-5 h-5" /></button>
            </div>
            <input 
              type="text" 
              value={newGroupName}
              onChange={(e) => setNewGroupName(e.target.value)}
              placeholder="输入小组名称" 
              className="w-full p-3 rounded-lg border border-slate-200 mb-6" 
            />
            <button onClick={handleCreateGroup} className="w-full p-3 rounded-lg bg-blue-600 text-white font-bold">创建</button>
          </div>
        </div>
      )}
    </div>
  );
}
