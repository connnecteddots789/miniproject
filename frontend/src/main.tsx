import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './style.css';

class AppErrorBoundary extends React.Component<React.PropsWithChildren, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidCatch(error: Error) { console.error('SENTINEL UI crashed:', error); }
  render() {
    if (this.state.failed) {
      return <div style={{ minHeight: '100vh', background: '#11100f', color: '#f4eee8', display: 'grid', placeItems: 'center', padding: 24, fontFamily: 'system-ui, sans-serif' }}>
        <div style={{ maxWidth: 520, border: '1px solid #65402b', borderRadius: 14, padding: 28, background: '#1b1816' }}>
          <h1 style={{ marginTop: 0 }}>SENTINEL recovered from a display error</h1>
          <p style={{ color: '#c5b8ad', lineHeight: 1.6 }}>The page hit an unexpected display error. Your backend experiment may still be running. Reload the interface to reconnect; completed research runs remain in Accuracy Lab history.</p>
          <button onClick={() => window.location.reload()} style={{ background: '#f47735', color: '#1b120d', border: 0, borderRadius: 7, padding: '11px 16px', fontWeight: 700, cursor: 'pointer' }}>Reload SENTINEL</button>
        </div>
      </div>;
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><AppErrorBoundary><App /></AppErrorBoundary></React.StrictMode>
);
