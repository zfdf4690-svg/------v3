import React, { useState } from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTheme } from './ThemeProvider';
import {
  LayoutDashboard,
  Video,
  Users,
  Database,
  Settings,
  Search,
  Bell,
  Sun,
  Moon,
  Sparkles,
  LogOut,
  Repeat,
  Shield
} from 'lucide-react';

export function Layout() {
  const location = useLocation();
  const { theme, setTheme } = useTheme();
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: '仪表盘' },
    { path: '/meetings', icon: Video, label: '我的会议' },
    { path: '/summaries', icon: Users, label: '会议总结' },
    { path: '/groups', icon: Users, label: '我的小组' },
    { path: '/knowledge-base', icon: Database, label: '知识库' },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 font-sans">
      {/* Sidebar */}
      <aside className="w-64 border-r border-blue-500/10 bg-white dark:bg-slate-900 flex flex-col h-full shrink-0">
        <div className="p-6 flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white">
            <Sparkles className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-base font-bold leading-none">AI 会议助手</h1>
            <p className="text-xs text-slate-500 mt-1">智能会议管理</p>
          </div>
        </div>
        
        <nav className="flex-1 px-4 py-4 flex flex-col gap-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg font-medium transition-colors ${
                  isActive
                    ? 'bg-blue-500/10 text-blue-600 dark:text-blue-400'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="text-sm">{item.label}</span>
              </Link>
            );
          })}
        </nav>
        
        <div className="p-4 border-t border-blue-500/10">
          <div className="bg-blue-500/5 rounded-xl p-4">
            <p className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wider mb-2">存储空间</p>
            <div className="h-1.5 w-full bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div className="h-full bg-blue-600 w-3/4"></div>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">已使用 7.5GB / 10GB</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col overflow-y-auto">
        {/* Topbar */}
        <header className="h-16 border-b border-blue-500/10 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md sticky top-0 z-10 flex items-center justify-between px-8">
          <div className="flex-1 max-w-xl">
            <div className="relative group">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 group-focus-within:text-blue-600 transition-colors" />
              <input
                type="text"
                placeholder="搜索会议、记录或文件..."
                className="w-full bg-slate-100 dark:bg-slate-800 border-none rounded-lg pl-10 pr-4 py-2 text-sm focus:ring-2 focus:ring-blue-500/20 transition-all outline-none"
              />
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button className="relative p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors">
              <Bell className="w-5 h-5" />
              <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-white dark:border-slate-900"></span>
            </button>
            <div className="h-8 w-[1px] bg-slate-200 dark:bg-slate-700 mx-2"></div>
            <div className="relative">
              <div className="flex items-center gap-3 cursor-pointer" onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}>
                <div className="text-right">
                  <p className="text-sm font-semibold">张经理</p>
                  <p className="text-[10px] text-slate-500">高级管理员</p>
                </div>
                <img
                  src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix"
                  alt="User Avatar"
                  className="w-10 h-10 rounded-full object-cover ring-2 ring-blue-500/10 bg-slate-100"
                />
              </div>
              {isUserMenuOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 p-2 z-50">
                  <button className="flex w-full items-center gap-2 px-4 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 rounded">
                    <Repeat className="w-4 h-4" /> 切换账号
                  </button>
                  <button className="flex w-full items-center gap-2 px-4 py-2 text-sm hover:bg-slate-100 dark:hover:bg-slate-700 rounded">
                    <Shield className="w-4 h-4" /> 隐私设置
                  </button>
                  <div className="my-1 border-t border-slate-100 dark:border-slate-700"></div>
                  <button className="flex w-full items-center gap-2 px-4 py-2 text-sm text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded">
                    <LogOut className="w-4 h-4" /> 退出登录
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Page Content */}
        <div className="flex-1">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
