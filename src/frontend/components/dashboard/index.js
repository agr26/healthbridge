import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { Activity, User, FileText, Bell } from 'lucide-react';

const PatientDashboard = ({ patientId }) => {
  const [activeTab, setActiveTab] = useState('overview');

  // Clinical status component with accessibility considerations
  const ClinicalStatus = ({ status, lastUpdated }) => (
    <div 
      role="status" 
      aria-live="polite"
      className="flex items-center space-x-2"
    >
      <span 
        className={`h-3 w-3 rounded-full ${
          status === 'critical' ? 'bg-red-500' :
          status === 'serious' ? 'bg-amber-500' :
          status === 'stable' ? 'bg-green-500' :
          'bg-blue-500'
        }`}
      />
      <span className="font-medium">{status}</span>
      <span className="text-sm text-gray-500">
        Updated {new Date(lastUpdated).toLocaleTimeString()}
      </span>
    </div>
  );

  // Vital signs display with color coding
  const VitalSigns = ({ vitals }) => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {vitals.map(vital => (
        <Card key={vital.id}>
          <CardContent className="p-4">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-500">{vital.name}</span>
              <span 
                className={`text-lg font-bold ${
                  vital.status === 'abnormal' ? 'text-red-600' : 'text-gray-900'
                }`}
              >
                {vital.value} {vital.unit}
              </span>
            </div>
            {vital.trend && (
              <LineChart width={200} height={50} data={vital.trend}>
                <Line 
                  type="monotone" 
                  dataKey="value" 
                  stroke="#4f46e5" 
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-8">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Patient Dashboard
            </h1>
            <p className="mt-2 text-gray-600">
              ID: {patientId} • DOB: 1985-06-15 • MRN: 123456
            </p>
          </div>
          <ClinicalStatus 
            status="stable"
            lastUpdated="2024-03-15T14:30:00"
          />
        </div>
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full max-w-2xl">
          <TabsTrigger value="overview">
            <Activity className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="vitals">
            <User className="w-4 h-4 mr-2" />
            Vitals
          </TabsTrigger>
          <TabsTrigger value="labs">
            <FileText className="w-4 h-4 mr-2" />
            Labs
          </TabsTrigger>
          <TabsTrigger value="alerts">
            <Bell className="w-4 h-4 mr-2" />
            Alerts
          </TabsTrigger>
        </TabsList>

        <div className="mt-6">
          <TabsContent value="overview">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Overview content */}
            </div>
          </TabsContent>

          <TabsContent value="vitals">
            <VitalSigns vitals={[
              { 
                id: 1, 
                name: 'Heart Rate', 
                value: 72, 
                unit: 'bpm',
                status: 'normal',
                trend: Array.from({ length: 10 }, (_, i) => ({
                  time: i,
                  value: 70 + Math.random() * 10
                }))
              },
              // Add more vital signs
            ]} />
          </TabsContent>

          {/* Add other tab contents */}
        </div>
      </Tabs>
    </div>
  );
};

export default PatientDashboard;