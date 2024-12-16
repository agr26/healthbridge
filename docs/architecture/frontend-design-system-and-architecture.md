# HealthBridge Frontend Design System

## 1. Design Principles

### 1.1 Core Principles
- **Clarity**: Clear presentation of medical information
- **Accessibility**: WCAG 2.1 AA compliance
- **Consistency**: Uniform patterns across the platform
- **Efficiency**: Minimize cognitive load for healthcare providers
- **Reliability**: Error prevention and clear feedback
- **Privacy**: Visual privacy considerations for sensitive data

### 1.2 Healthcare-Specific Considerations
- Color-blind friendly visualizations for medical data
- Clear hierarchy for critical patient information
- Quick-access patterns for emergency scenarios
- Mobile-responsive design for bedside use
- Fail-safe input patterns for medical data
- Clear status indicators for time-sensitive information

## 2. Color System

### 2.1 Primary Colors
```scss
$colors: (
  primary: (
    base: #2563EB,    // Main actions
    light: #93C5FD,   // Secondary elements
    dark: #1E40AF     // Emphasis
  ),
  critical: (
    error: #DC2626,   // Critical alerts
    warning: #F59E0B, // Warnings
    success: #10B981  // Positive outcomes
  ),
  neutral: (
    white: #FFFFFF,
    gray-100: #F3F4F6,
    gray-300: #D1D5DB,
    gray-500: #6B7280,
    gray-700: #374151,
    gray-900: #111827
  )
);
```

### 2.2 Clinical Status Colors
```scss
$clinical-status: (
  critical: #EF4444,
  serious: #F59E0B,
  stable: #10B981,
  recovered: #3B82F6
);
```