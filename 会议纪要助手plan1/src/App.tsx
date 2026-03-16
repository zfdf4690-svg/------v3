/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './components/ThemeProvider';
import { MeetingProvider } from './contexts/MeetingContext';
import { Toaster } from '../components/ui/sonner';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { MeetingPlanner } from './pages/MeetingPlanner';
import { ActiveMeeting } from './pages/ActiveMeeting';
import { MeetingSummaryDetail } from './pages/MeetingSummaryDetail';
import { MeetingSummaries } from './pages/MeetingSummaries';
import { KnowledgeBase } from './pages/KnowledgeBase';
import { MeetingsList } from './pages/MeetingsList';
import { MyGroups } from './pages/MyGroups';

export default function App() {
  return (
    <ThemeProvider defaultTheme="system" storageKey="vite-ui-theme">
      <MeetingProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="meetings" element={<MeetingsList />} />
              <Route path="meetings/new" element={<MeetingPlanner />} />
              <Route path="meetings/:id" element={<MeetingSummaryDetail />} />
              <Route path="summaries" element={<MeetingSummaries />} />
              <Route path="meetings/:id/active" element={<ActiveMeeting />} />
              <Route path="groups" element={<MyGroups />} />
              <Route path="knowledge-base" element={<KnowledgeBase />} />
            </Route>
          </Routes>
          <Toaster />
        </BrowserRouter>
      </MeetingProvider>
    </ThemeProvider>
  );
}
