import React from 'react';
import { ThemeProvider } from '@/components/theme-provider';
import { Toaster } from '@/components/ui/toaster';

// Design system components
const DesignSystem = {
  // Typography components
  Typography: {
    H1: ({ children, ...props }) => (
      <h1 className="text-4xl font-bold tracking-tight text-gray-900" {...props}>
        {children}
      </h1>
    ),
    H2: ({ children, ...props }) => (
      <h2 className="text-2xl font-semibold text-gray-800" {...props}>
        {children}
      </h2>
    ),
    Clinical: ({ value, status, ...props }) => (
      <span 
        className={`font-mono ${
          status === 'critical' ? 'text-red-600' :
          status === 'warning' ? 'text-amber-600' :
          'text-gray-900'
        }`}
        {...props}
      >
        {value}
      </span>
    )
  },

  // Clinical components
  Clinical: {
    VitalSign: ({ label, value, unit, range, isAbnormal }) => (
      <div className="p-3 border rounded-lg">
        <div className="text-sm text-gray-500">{label}</div>
        <div className="flex items-center space-x-2">
          <span className={`text-xl font-bold ${isAbnormal ? 'text-red-600' : 'text-gray-900'}`}>
            {value}
          </span>
          <span className="text-sm text-gray-500">{unit}</span>
          {isAbnormal && (
            <span className="px-2 py-1 text-xs text-red-600 bg-red-50 rounded-full">
              Outside range: {range}
            </span>
          )}
        </div>
      </div>
    ),

    LabResult: ({ name, value, unit, reference, status }) => (
      <div className="p-4 border-l-4 border-l-blue-500 bg-white shadow-sm">
        <div className="flex justify-between">
          <span className="font-medium">{name}</span>
          <span className="text-sm text-gray-500">{unit}</span>
        </div>
        <div className="mt-1 flex items-center space-x-2">
          <span className={`text-lg ${
            status === 'high' ? 'text-red-600' :
            status === 'low' ? 'text-amber-600' :
            'text-green-600'
          }`}>
            {value}
          </span>
          <span className="text-xs text-gray-500">({reference})</span>
        </div>
      </div>
    )
  },

  // Navigation components
  Navigation: {
    ClinicalBreadcrumb: ({ items }) => (
      <nav className="flex items-center space-x-2 text-sm">
        {items.map((item, index) => (
          <React.Fragment key={item.id}>
            {index > 0 && <span className="text-gray-400">/</span>}
            <a 
              href={item.href}
              className="text-gray-600 hover:text-gray-900"
            >
              {item.label}
            </a>
          </React.Fragment>
        ))}
      </nav>
    )
  }
};

export default DesignSystem;