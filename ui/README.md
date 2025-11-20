# Swing Trading UI

A modern React-based frontend for the Swing Trading API.

## Features

- **Dashboard**: View daily briefs and top trade recommendations
- **Recommendations**: Browse and explore detailed trade recommendations
- **Digital Twins**: Monitor and inspect digital twin models
- **Performance**: Analyze backtest and paper trading metrics

## Tech Stack

- React 18 with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- React Router for navigation
- Recharts for data visualization
- Axios for API communication

## Getting Started

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
npm install
```

### Development

Start the development server:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`.

### Build

Build for production:

```bash
npm run build
```

### Configuration

The API base URL can be configured via environment variable:

```bash
VITE_API_URL=http://localhost:8000 npm run dev
```

By default, the app expects the API to be running on `http://localhost:8000`.

## Project Structure

```
src/
├── api/          # API client and type definitions
├── components/   # Reusable UI components
├── layouts/      # Layout components
├── pages/        # Page components
└── App.tsx       # Main app with routing
```


