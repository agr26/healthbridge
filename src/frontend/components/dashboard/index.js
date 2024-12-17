import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  AreaChart, Area, BarChart, Bar
} from 'recharts';
import { 
  Activity, User, FileText, Bell, AlertTriangle, Heart, 
  Activity as VitalsIcon, ChevronUp, ChevronDown, Zap
} from 'lucide-react';

// WebSocket service
class WebSocketService {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.connections = new Map();
    this.callbacks = new Map();
  }

  connect(clientId, dataType, callback) {
    const ws = new WebSocket(`${this.baseUrl}/ws/${clientId}/${dataType}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (callback) callback(data);
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error: ${error}`);
    };

    ws.onclose = () => {
      this.reconnect(clientId, dataType, callback);
    };

    this.connections.set(dataType, ws);
    this.callbacks.set(dataType, callback);
  }

  reconnect(clientId, dataType, callback) {
    setTimeout(() => {
      this.connect(clientId, dataType, callback);
    }, 5000);
  }

  disconnect(dataType) {
    if (this.connections.has(dataType)) {
      this.connections.get(dataType).close();
      this.connections.delete(dataType);
      this.callbacks.delete(dataType);
    }
  }
}

// API service
const API_BASE_URL = 'http://localhost:8000/api';
const WS_BASE_URL = 'ws://localhost:8000';

const wsService = new WebSocketService(WS_BASE_URL);

const fetchPatientData = async (patientId) => {
  const response = await fetch(`${API_BASE_URL}/patient/${patientId}`);
  if (!response.ok) throw new Error('Failed to fetch patient data');
  return response.json();
};

const fetchPatientRisk = async (patientId) => {
  const response = await fetch(`${API_BASE_URL}/patient/${patientId}/risk`);
  if (!response.ok) throw new Error('Failed to fetch risk assessment');
  return response.json();
};

const PatientDashboard = ({ patientId }) => {
  // State management
  const [patientData, setPatientData] = useState(null);
  const [riskProfile, setRiskProfile] = useState(null);
  const [vitals, setVitals] = useState([]);
  const [labs, setLabs] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [realTimeData, setRealTimeData] = useState({
    vitals: [],
    predictions: [],
    equity: null,
    quality: null
  });

  // Connect to WebSocket streams
  useEffect(() => {
    wsService.connect(patientId, 'vitals', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        vitals: [...prev.vitals.slice(-19), data]
      }));
    });

    wsService.connect(patientId, 'predictions', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        predictions: [...prev.predictions.slice(-19), data]
      }));
    });

    wsService.connect(patientId, 'equity', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        equity: data
      }));
    });

    wsService.connect(patientId, 'quality', (data) => {
      setRealTimeData(prev => ({
        ...prev,
        quality: data
      }));
    });

    return () => {
      ['vitals', 'predictions', 'equity', 'quality'].forEach(type => {
        wsService.disconnect(type);
      });
    };
  }, [patientId]);

  // Fetch initial data
  useEffect(() => {
    const loadPatientData = async () => {
      try {
        setLoading(true);
        const [data, risk] = await Promise.all([
          fetchPatientData(patientId),
          fetchPatientRisk(patientId)
        ]);
        setPatientData(data);
        setRiskProfile(risk);
        setVitals(data.vitals || []);
        setLabs(data.labs || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    loadPatientData();
  }, [patientId]);

  // Clinical Status Component
  const ClinicalStatus = ({ status, lastUpdated }) => (
    <div 
      role="status" 
      aria-live="polite"
      className="flex items-center space-x-2 p-4 rounded-lg bg-white shadow-sm"
    >
      <span 
        className={`h-3 w-3 rounded-full ${
          status === 'critical' ? 'bg-red-500' :
          status === 'serious' ? 'bg-amber-500' :
          status === 'stable' ? 'bg-green-500' :
          'bg-blue-500'
        }`}
        aria-hidden="true"
      />
      <span className="font-medium">{status}</span>
      <span className="text-sm text-gray-500">
        Updated {new Date(lastUpdated).toLocaleTimeString()}
      </span>
    </div>
  );

  // Risk Assessment Component
  const RiskAssessment = ({ risk, realTimeData }) => (
    <Card className="bg-white shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className={
            risk.risk_level === 'High' ? 'text-red-500' :
            risk.risk_level === 'Medium' ? 'text-amber-500' :
            'text-green-500'
          } />
          Risk Assessment
          {realTimeData?.predictions?.length > 0 && (
            <Badge variant="outline" className="ml-2">
              Real-time
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <span>Risk Level:</span>
            <span className={`font-bold ${
              risk.risk_level === 'High' ? 'text-red-500' :
              risk.risk_level === 'Medium' ? 'text-amber-500' :
              'text-green-500'
            }`}>
              {risk.risk_level}
            </span>
          </div>
          
          {realTimeData?.predictions?.length > 0 && (
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={realTimeData.predictions}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis domain={[0, 1]} />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="riskScore" 
                    stroke="#8884d8" 
                    name="Risk Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div>
            <h4 className="font-medium mb-2">Risk Factors:</h4>
            <ul className="list-disc pl-5 space-y-1">
              {risk.risk_factors.map((factor, index) => (
                <li key={index}>{factor}</li>
              ))}
            </ul>
          </div>
          
          {risk.recommendations && (
            <div>
              <h4 className="font-medium mb-2">Recommendations:</h4>
              <ul className="list-disc pl-5 space-y-1">
                {risk.recommendations.map((rec, index) => (
                  <li key={index}>{rec}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );

  // Vital Signs Component
  const VitalSigns = ({ vitals, realTimeData }) => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {vitals.map(vital => (
          <Card key={vital.id} className="bg-white shadow-sm">
            <CardContent className="p-4">
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-500">{vital.name}</span>
                  <span className={`text-lg font-bold ${
                    vital.status === 'abnormal' ? 'text-red-600' : 'text-gray-900'
                  }`}>
                    {vital.value} {vital.unit}
                  </span>
                </div>
                {(vital.trend || realTimeData?.vitals) && (
                  <ResponsiveContainer width="100%" height={100}>
                    <LineChart data={realTimeData?.vitals || vital.trend}>
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#4f46e5" 
                        strokeWidth={2}
                        dot={false}
                      />
                      <XAxis dataKey="time" />
                      <YAxis domain={['auto', 'auto']} />
                      <Tooltip />
                    </LineChart>
                  </ResponsiveContainer>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {realTimeData?.vitals?.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Real-time Vital Trends</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={realTimeData.vitals}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {Object.keys(realTimeData.vitals[0] || {})
                    .filter(key => key !== 'timestamp')
                    .map((key, index) => (
                      <Line 
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={`hsl(${index * 45}, 70%, 50%)`}
                        name={key}
                      />
                    ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  // Lab Results Component
  const LabResults = ({ labs }) => (
    <div className="space-y-4">
      {labs.map(lab => (
        <Card key={lab.id}>
          <CardContent className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="font-medium">{lab.name}</h3>
                <p className="text-sm text-gray-500">{lab.category}</p>
              </div>
              <div className="text-right">
                <span className={`text-lg font-bold ${
                  lab.isAbnormal ? 'text-red-600' : 'text-gray-900'
                }`}>
                  {lab.value} {lab.unit}
                </span>
                <p className="text-sm text-gray-500">
                  Reference: {lab.referenceRange}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );

  // Alerts Component
  const AlertsPanel = ({ alerts, realTimeData }) => (
    <div className="space-y-4">
      {alerts.map(alert => (
        <Alert
          key={alert.id}
          variant={alert.severity === 'high' ? 'destructive' : 'default'}
        >
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{alert.message}</AlertDescription>
        </Alert>
      ))}

      {realTimeData?.quality && (
        <Card>
          <CardHeader>
            <CardTitle>Quality Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[realTimeData.quality]}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="department" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="readmissionRate" fill="#8884d8" name="Readmission Rate" />
                  <Bar dataKey="mortalityRate" fill="#82ca9d" name="Mortality Rate" />
                  <Bar dataKey="satisfaction" fill="#ffc658" name="Satisfaction" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  // Loading and Error States
  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>
          Error loading patient data: {error}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <header className="mb-8">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Patient Dashboard
            </h1>
            <p className="mt-2 text-gray-600">
              ID: {patientId} • {patientData?.demographics?.dateOfBirth} • MRN: {patientData?.demographics?.mrn}
            </p>
          </div>
          <ClinicalStatus 
            status={patientData?.clinicalStatus?.current}
            lastUpdated={patientData?.clinicalStatus?.lastUpdated}
          />
        </div>

        {realTimeData?.equity && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-blue-900">Health Equity Insights</h2>
              <Badge variant="outline" className="ml-2">Real-time</Badge>
            </div>
            <div className="mt-2 grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(realTimeData.equity.metrics).map(([key, value]) => (
                <div key={key} className="p-3 bg-white rounded-md shadow-sm">
                  <span className="text-sm text-gray-500">{key}</span>
                  <div className="mt-1 font-semibold">{value.toFixed(2)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </header>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid grid-cols-4 w-full max-w-2xl">
          <TabsTrigger value="overview">
            <Activity className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="vitals">
            <Heart className="w-4 h-4 mr-2" />
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
              {riskProfile && (
                <RiskAssessment 
                  risk={riskProfile} 
                  realTimeData={realTimeData}
                />
              )}
              
              <Card>
                <CardHeader>
                  <CardTitle>Key Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {realTimeData?.predictions?.slice(-1)?.[0]?.metrics?.map((metric, index) => (
                      <div 
                        key={index}
                        className="flex items-center justify-between p-2 bg-gray-50 rounded"
                      >
                        <span>{metric.name}</span>
                        <div className="flex items-center">
                          <span className={`font-semibold ${
                            metric.trend === 'up' ? 'text-green-600' :
                            metric.trend === 'down' ? 'text-red-600' :
                            'text-gray-600'
                          }`}>
                            {metric.value}
                          </span>
                          {metric.trend && (
                            <span className="ml-2">
                              {metric.trend === 'up' ? <ChevronUp className="w-4 h-4 text-green-600" /> :
                               metric.trend === 'down' ? <ChevronDown className="w-4 h-4 text-red-600" /> : null}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="vitals">
            <VitalSigns 
              vitals={vitals} 
              realTimeData={realTimeData}
            />
          </TabsContent>

          <TabsContent value="labs">
            <LabResults labs={labs} />
          </TabsContent>

          <TabsContent value="alerts">
            <AlertsPanel 
              alerts={alerts}
              realTimeData={realTimeData}
            />
          </TabsContent>
        </div>
      </Tabs>

      {/* Real-time Status Indicator */}
      <div className="fixed bottom-4 right-4">
        <div className="flex items-center space-x-2 bg-white p-2 rounded-lg shadow-lg">
          <div className={`h-2 w-2 rounded-full ${
            Object.values(realTimeData).some(v => v?.length > 0 || v !== null)
              ? 'bg-green-500 animate-pulse'
              : 'bg-gray-500'
          }`} />
          <span className="text-sm text-gray-600">
            {Object.values(realTimeData).some(v => v?.length > 0 || v !== null)
              ? 'Real-time updates active'
              : 'Connecting...'}
          </span>
        </div>
      </div>
    </div>
  );
};

export default PatientDashboard;