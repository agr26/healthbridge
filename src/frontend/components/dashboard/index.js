import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, User, FileText, Bell, AlertTriangle } from 'lucide-react';

// API service
const API_BASE_URL = 'http://localhost:8000/api';

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
  const [activeTab, setActiveTab] = useState('overview');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch data on component mount
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

  // Clinical status component with accessibility
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
  const RiskAssessment = ({ risk }) => (
    <Card className="bg-white shadow-sm">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className={
            risk.risk_level === 'High' ? 'text-red-500' :
            risk.risk_level === 'Medium' ? 'text-amber-500' :
            'text-green-500'
          } />
          Risk Assessment
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
  const VitalSigns = ({ vitals }) => (
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
              {vital.trend && (
                <ResponsiveContainer width="100%" height={100}>
                  <LineChart data={vital.trend}>
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
              {riskProfile && <RiskAssessment risk={riskProfile} />}
              {/* Additional overview components */}
            </div>
          </TabsContent>

          <TabsContent value="vitals">
            <VitalSigns vitals={vitals} />
          </TabsContent>

          <TabsContent value="labs">
            {/* Labs content */}
          </TabsContent>

          <TabsContent value="alerts">
            {/* Alerts content */}
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
};

export default PatientDashboard;