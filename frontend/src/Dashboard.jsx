import React, { useState, useEffect } from 'react';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import {
    Baby, Bot, Activity, Calendar, Award, AlertTriangle, ChevronRight,
    Volume2, Plus, X, Save, Clock, User, PlusCircle, Trash2, ArrowLeft,
    Heart, Pill, Shield, FileText, MessageSquare, ChevronDown, ChevronUp,
    Stethoscope, Users as UsersIcon, Loader
} from 'lucide-react';

const BACKEND = import.meta.env.VITE_API_URL;

const Dashboard = ({ user, onBack, onEmergencyCall }) => {
    // ─── State ──────────────────────────────────────────────────────────
    const [loading, setLoading] = useState(true);
    const [dashData, setDashData] = useState(null);
    const [doctorSummary, setDoctorSummary] = useState(null);
    const [familySummary, setFamilySummary] = useState(null);
    const [chatHistory, setChatHistory] = useState([]);
    const [showHistory, setShowHistory] = useState(false);
    const [historyPage, setHistoryPage] = useState(1);
    const [historyTotal, setHistoryTotal] = useState(0);
    const [activeTab, setActiveTab] = useState('overview'); // overview | doctor | family
    const [expandedInteraction, setExpandedInteraction] = useState(null);

    // Keep existing features
    const [stats, setStats] = useState(() => {
        const saved = localStorage.getItem(`stats_${user?.email || user?.phoneNumber}`);
        return saved ? JSON.parse(saved) : {
            kicks: 0, sleep: 8, appointments: [
                { id: 1, date: '2026-02-25', time: '10:00 AM', title: 'Routine Check-up', location: 'District Hospital' }
            ]
        };
    });
    const [isAddingAppointment, setIsAddingAppointment] = useState(false);
    const [newAppointment, setNewAppointment] = useState({ title: '', date: '', time: '', location: '' });

    // ─── User Identifier ────────────────────────────────────────────────
    const identifier = user?.phoneNumber || user?.email || '';

    // ─── Pregnancy Calculations ─────────────────────────────────────────
    const pregnancyDate = user?.pregnancyDate || '';
    const currentWeek = (() => {
        if (!pregnancyDate) return 0;
        const diff = new Date() - new Date(pregnancyDate);
        return Math.floor(diff / (1000 * 60 * 60 * 24 * 7));
    })();
    const progressPercent = Math.min((currentWeek / 40) * 100, 100);
    const trimester = currentWeek <= 12 ? 1 : (currentWeek <= 26 ? 2 : 3);
    const getBabySize = (w) => {
        if (w < 5) return "Sesame Seed"; if (w < 10) return "Grape"; if (w < 15) return "Lemon";
        if (w < 20) return "Banana"; if (w < 25) return "Corn Ear"; if (w < 30) return "Cabbage";
        if (w < 35) return "Pineapple"; return "Watermelon";
    };

    // ─── Fetch Dashboard Data ───────────────────────────────────────────
    useEffect(() => {
        if (!identifier) { setLoading(false); return; }
        const fetchAll = async () => {
            try {
                const [dashRes, docRes, famRes] = await Promise.all([
                    fetch(`${BACKEND}/api/dashboard/${encodeURIComponent(identifier)}`),
                    fetch(`${BACKEND}/api/dashboard/${encodeURIComponent(identifier)}/summary/doctor`),
                    fetch(`${BACKEND}/api/dashboard/${encodeURIComponent(identifier)}/summary/family`)
                ]);
                const dashJson = await dashRes.json();
                const docJson = await docRes.json();
                const famJson = await famRes.json();
                setDashData(dashJson.data || dashJson);
                setDoctorSummary(docJson.summary);
                setFamilySummary(famJson.summary);
            } catch (e) { console.error('Dashboard fetch error:', e); }
            setLoading(false);
        };
        fetchAll();
    }, [identifier]);

    // ─── Fetch Chat History ─────────────────────────────────────────────
    const loadHistory = async (page = 1) => {
        try {
            const res = await fetch(`${BACKEND}/api/dashboard/${encodeURIComponent(identifier)}/history?page=${page}&limit=20`);
            const data = await res.json();
            setChatHistory(prev => page === 1 ? data.history : [...prev, ...data.history]);
            setHistoryTotal(data.pagination?.total || 0);
            setHistoryPage(page);
        } catch (e) { console.error('History fetch error:', e); }
    };

    // ─── Persist Local Data ─────────────────────────────────────────────
    useEffect(() => {
        if (identifier) localStorage.setItem(`stats_${identifier}`, JSON.stringify(stats));
    }, [stats, identifier]);

    const handleAddAppointment = (e) => {
        e.preventDefault();
        setStats(prev => ({ ...prev, appointments: [...prev.appointments, { ...newAppointment, id: Date.now() }].sort((a, b) => new Date(a.date) - new Date(b.date)) }));
        setIsAddingAppointment(false);
        setNewAppointment({ title: '', date: '', time: '', location: '' });
    };

    // ─── Card Style ─────────────────────────────────────────────────────
    const card = { background: 'white', padding: '2rem', borderRadius: '24px', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', border: '1px solid #f1f5f9' };
    const badge = (color) => ({ display: 'inline-block', padding: '0.3rem 0.8rem', borderRadius: '50px', fontSize: '0.75rem', fontWeight: '700', background: color + '15', color });

    if (loading) return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '100vh' }}>
            <Loader className="spin" size={40} color="var(--primary)" />
            <p style={{ marginLeft: '1rem', color: 'var(--primary)' }}>Loading your health data...</p>
        </div>
    );

    const symptoms = dashData?.symptoms || [];
    const medications = dashData?.medications || [];
    const recentInteractions = dashData?.recentInteractions || [];
    const statsData = dashData?.stats || {};

    return (
        <Motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ padding: '100px 0', minHeight: '100vh', background: '#f8fafc' }}>
            <div className="container">
                {/* ─── Header ────────────────────────────────────────── */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
                    <div>
                        <h1 style={{ fontSize: '2.2rem', color: 'var(--text-dark)', margin: 0 }}>Namaste, {user?.name || 'Mataji'}!</h1>
                        <p style={{ color: 'var(--text-light)', fontSize: '1.1rem', margin: '0.3rem 0 0' }}>
                            {statsData.totalInteractions > 0
                                ? `${statsData.totalInteractions} conversations tracked • Week ${currentWeek}`
                                : `Fizician is tracking your Week ${currentWeek} journey`}
                        </p>
                    </div>
                    <button onClick={onBack} className="btn-primary" style={{ background: 'transparent', border: '2px solid var(--primary)', color: 'var(--primary)', padding: '0.7rem 1.5rem' }}>
                        <ArrowLeft size={16} /> Back to Chat
                    </button>
                </div>

                {/* ─── Summary Tabs ──────────────────────────────────── */}
                <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '2rem', background: '#f1f5f9', borderRadius: '16px', padding: '0.4rem' }}>
                    {[
                        { key: 'overview', label: 'Health Overview', icon: <Activity size={16} /> },
                        { key: 'doctor', label: 'Doctor Summary', icon: <Stethoscope size={16} /> },
                        { key: 'family', label: 'Family Summary', icon: <UsersIcon size={16} /> }
                    ].map(tab => (
                        <button key={tab.key} onClick={() => setActiveTab(tab.key)} style={{
                            flex: 1, padding: '0.8rem', border: 'none', borderRadius: '12px', cursor: 'pointer', fontWeight: '700', fontSize: '0.9rem',
                            background: activeTab === tab.key ? 'white' : 'transparent',
                            color: activeTab === tab.key ? 'var(--primary)' : '#64748b',
                            boxShadow: activeTab === tab.key ? '0 2px 8px rgba(0,0,0,0.08)' : 'none',
                            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', transition: 'all 0.3s'
                        }}>{tab.icon}{tab.label}</button>
                    ))}
                </div>

                {/* ─── TAB: OVERVIEW ─────────────────────────────────── */}
                {activeTab === 'overview' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>

                        {/* Pregnancy Progress */}
                        <Motion.div whileHover={{ y: -3 }} style={card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                                <div style={{ background: '#E0F2FE', width: '48px', height: '48px', borderRadius: '14px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}><Baby color="var(--primary)" size={26} /></div>
                                <div style={{ textAlign: 'right' }}>
                                    <span style={badge('var(--primary)')}>Week {currentWeek}</span>
                                    <small style={{ display: 'block', color: '#94a3b8', marginTop: '0.3rem' }}>Trimester {trimester}</small>
                                </div>
                            </div>
                            <h3 style={{ fontSize: '1.3rem', marginBottom: '0.8rem' }}>Pregnancy Progress</h3>
                            <div style={{ height: '10px', background: '#f1f5f9', borderRadius: '10px', overflow: 'hidden', marginBottom: '1rem' }}>
                                <Motion.div initial={{ width: 0 }} animate={{ width: `${progressPercent}%` }} transition={{ duration: 1.5 }}
                                    style={{ height: '100%', background: 'linear-gradient(90deg, var(--primary), var(--secondary))' }} />
                            </div>
                            <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>
                                Baby is size of a <b style={{ color: 'var(--primary)' }}>{getBabySize(currentWeek)}</b> • <b style={{ color: 'var(--primary)' }}>{Math.max(0, 40 - currentWeek)}</b> weeks to go!
                            </p>
                        </Motion.div>

                        {/* Stats Cards */}
                        <div style={card}>
                            <h3 style={{ fontSize: '1.3rem', marginBottom: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Activity color="var(--primary)" size={22} /> Health Stats</h3>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                                {[
                                    { label: 'Total Interactions', value: statsData.totalInteractions || 0, color: 'var(--primary)' },
                                    { label: 'Avg Severity', value: `${statsData.avgSeverity || 0}/10`, color: (statsData.avgSeverity || 0) > 5 ? '#ef4444' : '#10b981' },
                                    { label: 'Relief Rate', value: `${statsData.reliefRate || 0}%`, color: '#10b981' },
                                    { label: 'Symptoms Tracked', value: symptoms.length, color: '#f59e0b' }
                                ].map((s, i) => (
                                    <div key={i} style={{ padding: '1rem', background: '#f8fafc', borderRadius: '16px', textAlign: 'center' }}>
                                        <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>{s.label}</p>
                                        <h4 style={{ fontSize: '1.4rem', color: s.color, margin: '0.3rem 0 0' }}>{s.value}</h4>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Symptoms Timeline */}
                        <div style={card}>
                            <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Heart color="#ef4444" size={22} /> Symptom Tracker</h3>
                            {symptoms.length === 0 ? (
                                <p style={{ color: '#94a3b8', textAlign: 'center', padding: '2rem 0' }}>No symptoms recorded yet. Talk to Fizician!</p>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', maxHeight: '280px', overflowY: 'auto' }}>
                                    {symptoms.slice(0, 8).map((s, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.8rem 1rem', background: '#f8fafc', borderRadius: '14px' }}>
                                            <div>
                                                <strong style={{ fontSize: '0.95rem', textTransform: 'capitalize' }}>{s.name}</strong>
                                                <small style={{ display: 'block', color: '#94a3b8' }}>{s.occurrences}x mentioned</small>
                                            </div>
                                            <span style={badge(s.status === 'active' ? '#ef4444' : s.status === 'relieved' ? '#10b981' : '#f59e0b')}>{s.status}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Medications */}
                        <div style={card}>
                            <h3 style={{ fontSize: '1.3rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}><Pill color="#8b5cf6" size={22} /> Medication Tracker</h3>
                            {medications.length === 0 ? (
                                <p style={{ color: '#94a3b8', textAlign: 'center', padding: '2rem 0' }}>No medications recorded yet.</p>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem', maxHeight: '280px', overflowY: 'auto' }}>
                                    {medications.slice(0, 8).map((m, i) => (
                                        <div key={i} style={{ padding: '0.8rem 1rem', background: '#f8fafc', borderRadius: '14px' }}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <strong style={{ textTransform: 'capitalize' }}>{m.name}</strong>
                                                <span style={badge(m.timesTaken > m.timesSkipped ? '#10b981' : '#ef4444')}>
                                                    {m.timesTaken}✓ / {m.timesSkipped}✗
                                                </span>
                                            </div>
                                            {m.effects.length > 0 && <small style={{ color: '#64748b' }}>Effect: {m.effects[m.effects.length - 1]}</small>}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Appointments */}
                        <div style={card}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                                <h3 style={{ fontSize: '1.3rem', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}><Calendar color="var(--primary)" size={22} /> Appointments</h3>
                                <button onClick={() => setIsAddingAppointment(true)} style={{ background: 'none', border: 'none', color: 'var(--primary)', cursor: 'pointer' }}><PlusCircle size={24} /></button>
                            </div>
                            <div style={{ maxHeight: '200px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                                {stats.appointments.map(a => (
                                    <div key={a.id} style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', padding: '0.8rem', background: '#f8fafc', borderRadius: '14px' }}>
                                        <div style={{ background: 'white', padding: '0.4rem 0.6rem', borderRadius: '10px', textAlign: 'center', border: '1px solid #e2e8f0', minWidth: '50px' }}>
                                            <small style={{ display: 'block', fontSize: '0.6rem', color: 'var(--primary)', fontWeight: 'bold' }}>{new Date(a.date).toLocaleString('default', { month: 'short' }).toUpperCase()}</small>
                                            <span style={{ fontWeight: 'bold' }}>{new Date(a.date).getDate()}</span>
                                        </div>
                                        <div style={{ flex: 1 }}><h4 style={{ fontSize: '0.9rem', margin: 0 }}>{a.title}</h4><p style={{ fontSize: '0.75rem', color: '#64748b', margin: 0 }}>{a.time} • {a.location}</p></div>
                                        <button onClick={() => setStats(p => ({ ...p, appointments: p.appointments.filter(x => x.id !== a.id) }))} style={{ color: '#cbd5e1', background: 'none', border: 'none', cursor: 'pointer' }}><Trash2 size={14} /></button>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Emergency */}
                        <div style={{ ...card, background: '#fef2f2', border: '2px solid #fee2e2', textAlign: 'center' }}>
                            <div style={{ width: '56px', height: '56px', background: '#ef4444', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', margin: '0 auto 1rem', boxShadow: '0 0 20px rgba(239,68,68,0.3)' }}><AlertTriangle color="white" size={28} /></div>
                            <h3 style={{ color: '#b91c1c', marginBottom: '0.5rem' }}>Need Help Now?</h3>
                            <button onClick={onEmergencyCall} style={{ background: '#ef4444', color: 'white', border: 'none', padding: '0.8rem 2rem', borderRadius: '50px', fontWeight: 'bold', cursor: 'pointer' }}>EMERGENCY CALL</button>
                        </div>

                        {/* Recent Interactions + Read More */}
                        <div style={{ ...card, gridColumn: '1 / -1' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                                <h3 style={{ fontSize: '1.3rem', display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}><MessageSquare color="var(--primary)" size={22} /> Recent Conversations</h3>
                                <button onClick={() => { setShowHistory(!showHistory); if (!showHistory && chatHistory.length === 0) loadHistory(1); }}
                                    style={{ background: 'var(--primary)', color: 'white', border: 'none', padding: '0.6rem 1.2rem', borderRadius: '50px', fontSize: '0.85rem', fontWeight: '600', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                    {showHistory ? <><ChevronUp size={16} /> Hide</> : <><FileText size={16} /> Read More</>}
                                </button>
                            </div>

                            {recentInteractions.length === 0 ? (
                                <p style={{ color: '#94a3b8', textAlign: 'center', padding: '1.5rem 0' }}>No conversations yet. Start chatting with Fizician!</p>
                            ) : (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', maxHeight: showHistory ? 'none' : '300px', overflowY: 'auto' }}>
                                    {(showHistory ? chatHistory : recentInteractions.slice(0, 5)).map((item, i) => (
                                        <div key={item.id || i} style={{ padding: '1rem', background: '#f8fafc', borderRadius: '14px', cursor: 'pointer', transition: 'all 0.2s', border: expandedInteraction === i ? '2px solid var(--primary)' : '2px solid transparent' }}
                                            onClick={() => setExpandedInteraction(expandedInteraction === i ? null : i)}>
                                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                <div style={{ flex: 1 }}>
                                                    <p style={{ margin: 0, fontWeight: '600', fontSize: '0.9rem' }}>Patient: {(item.userMessage || item.userMessageEnglish || '').slice(0, 80)}{(item.userMessage || item.userMessageEnglish || '').length > 80 ? '...' : ''}</p>
                                                    <small style={{ color: '#64748b' }}>{new Date(item.timestamp).toLocaleString()}</small>
                                                </div>
                                                <div style={{ display: 'flex', gap: '0.3rem', alignItems: 'center' }}>
                                                    {item.severity > 5 && <span style={badge('#ef4444')}>Sev: {item.severity}</span>}
                                                    {expandedInteraction === i ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                                </div>
                                            </div>
                                            {expandedInteraction === i && (
                                                <Motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} style={{ marginTop: '0.8rem', borderTop: '1px solid #e2e8f0', paddingTop: '0.8rem' }}>
                                                    <p style={{ margin: '0 0 0.5rem', color: '#10b981' }}>Fizician: {item.aiReply || item.aiReplyEnglish || 'No reply recorded'}</p>
                                                    {(item.symptoms || []).length > 0 && <p style={{ margin: 0, fontSize: '0.85rem' }}>Symptoms: {item.symptoms.map(s => s.name).join(', ')}</p>}
                                                    {item.aiSummary && <p style={{ margin: '0.3rem 0 0', fontSize: '0.85rem', color: '#8b5cf6' }}>Summary: {item.aiSummary}</p>}
                                                </Motion.div>
                                            )}
                                        </div>
                                    ))}
                                    {showHistory && chatHistory.length < historyTotal && (
                                        <button onClick={() => loadHistory(historyPage + 1)} style={{ margin: '0.5rem auto', background: 'var(--primary)', color: 'white', border: 'none', padding: '0.6rem 1.5rem', borderRadius: '50px', cursor: 'pointer', fontWeight: '600' }}>Load More</button>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* ─── TAB: DOCTOR SUMMARY ───────────────────────────── */}
                {activeTab === 'doctor' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
                        <div style={{ ...card, gridColumn: '1 / -1', borderLeft: '4px solid var(--primary)' }}>
                            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}><Stethoscope color="var(--primary)" size={24} /> Doctor's Review Summary</h3>
                            {typeof doctorSummary === 'string' ? <p style={{ color: '#64748b' }}>{doctorSummary}</p> : doctorSummary && (
                                <>
                                    {(doctorSummary.redFlags || []).length > 0 && (
                                        <div style={{ background: '#fef2f2', padding: '1rem', borderRadius: '14px', marginBottom: '1rem' }}>
                                            <h4 style={{ color: '#b91c1c', margin: '0 0 0.5rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}><AlertTriangle size={18} /> Red Flags</h4>
                                            <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#991b1b' }}>{doctorSummary.redFlags.map((f, i) => <li key={i} style={{ marginBottom: '0.3rem' }}>{f}</li>)}</ul>
                                        </div>
                                    )}
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                                        <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '14px' }}>
                                            <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>High Severity Events</p>
                                            <h4 style={{ color: '#ef4444', margin: '0.3rem 0 0', fontSize: '1.5rem' }}>{doctorSummary.highSeverityEvents || 0}</h4>
                                        </div>
                                        <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '14px' }}>
                                            <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>Total Interactions</p>
                                            <h4 style={{ color: 'var(--primary)', margin: '0.3rem 0 0', fontSize: '1.5rem' }}>{doctorSummary.totalInteractions || 0}</h4>
                                        </div>
                                    </div>
                                    {(doctorSummary.activeSymptoms || []).length > 0 && (
                                        <div style={{ marginBottom: '1rem' }}>
                                            <h4 style={{ marginBottom: '0.5rem' }}>Active Symptoms</h4>
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>{doctorSummary.activeSymptoms.map((s, i) => <span key={i} style={badge('#ef4444')}>{s}</span>)}</div>
                                        </div>
                                    )}
                                    {(doctorSummary.skippedMedications || []).length > 0 && (
                                        <div style={{ marginBottom: '1rem' }}>
                                            <h4 style={{ marginBottom: '0.5rem' }}>Skipped Medications</h4>
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>{doctorSummary.skippedMedications.map((m, i) => <span key={i} style={badge('#f59e0b')}>{m}</span>)}</div>
                                        </div>
                                    )}
                                    {doctorSummary.doctorNotes && (
                                        <div style={{ background: '#f0fdf4', padding: '1rem', borderRadius: '14px' }}>
                                            <h4 style={{ margin: '0 0 0.5rem', color: '#166534' }}>AI Clinical Notes</h4>
                                            <pre style={{ whiteSpace: 'pre-wrap', fontSize: '0.85rem', color: '#374151', margin: 0, fontFamily: 'inherit' }}>{doctorSummary.doctorNotes}</pre>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                )}

                {/* ─── TAB: FAMILY SUMMARY ───────────────────────────── */}
                {activeTab === 'family' && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.5rem' }}>
                        <div style={{ ...card, gridColumn: '1 / -1', borderLeft: '4px solid #10b981' }}>
                            <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}><UsersIcon color="#10b981" size={24} /> Family Health Summary</h3>
                            {typeof familySummary === 'string' ? <p style={{ color: '#64748b' }}>{familySummary}</p> : familySummary && (
                                <>
                                    <div style={{ background: familySummary.overallHealth === 'Good' ? '#f0fdf4' : familySummary.overallHealth === 'Needs Attention' ? '#fffbeb' : '#fef2f2', padding: '1.5rem', borderRadius: '16px', marginBottom: '1.5rem', textAlign: 'center' }}>
                                        <h2 style={{ margin: 0, color: familySummary.overallHealth === 'Good' ? '#166534' : familySummary.overallHealth === 'Needs Attention' ? '#92400e' : '#991b1b', fontSize: '1.8rem' }}>
                                            {familySummary.overallHealth === 'Good' ? 'Good' : familySummary.overallHealth === 'Needs Attention' ? 'Needs Attention' : 'Needs Care'} — {familySummary.overallHealth}
                                        </h2>
                                        <p style={{ margin: '0.5rem 0 0', color: '#64748b' }}>{familySummary.message}</p>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                                        <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '14px', textAlign: 'center' }}>
                                            <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>Relief Times</p>
                                            <h4 style={{ color: '#10b981', margin: '0.3rem 0 0' }}>{familySummary.reliefOccurrences || 0}</h4>
                                        </div>
                                        <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '14px', textAlign: 'center' }}>
                                            <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>Recent Chats</p>
                                            <h4 style={{ color: 'var(--primary)', margin: '0.3rem 0 0' }}>{familySummary.totalRecentChats || 0}</h4>
                                        </div>
                                        <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '14px', textAlign: 'center' }}>
                                            <p style={{ color: '#94a3b8', fontSize: '0.8rem', margin: 0 }}>Avg Severity</p>
                                            <h4 style={{ color: (familySummary.averageSeverity || 0) > 5 ? '#ef4444' : '#10b981', margin: '0.3rem 0 0' }}>{familySummary.averageSeverity || 0}/10</h4>
                                        </div>
                                    </div>
                                    {(familySummary.recentSymptoms || []).length > 0 && (
                                        <div style={{ marginBottom: '1rem' }}><h4 style={{ marginBottom: '0.5rem' }}>Recent Symptoms</h4><div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>{familySummary.recentSymptoms.map((s, i) => <span key={i} style={badge('#f59e0b')}>{s}</span>)}</div></div>
                                    )}
                                    {(familySummary.medicationsTaken || []).length > 0 && (
                                        <div><h4 style={{ marginBottom: '0.5rem' }}>Medications Being Taken</h4><div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>{familySummary.medicationsTaken.map((m, i) => <span key={i} style={badge('#10b981')}>{m}</span>)}</div></div>
                                    )}
                                </>
                            )}
                        </div>
                    </div>
                )}
            </div>

            {/* ─── APPOINTMENT MODAL ─────────────────────────────────── */}
            <AnimatePresence>
                {isAddingAppointment && (
                    <Motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                        style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(5px)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 2000 }}
                        onClick={() => setIsAddingAppointment(false)}>
                        <Motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} exit={{ scale: 0.9 }}
                            style={{ background: 'white', padding: '2rem', borderRadius: '24px', width: '90%', maxWidth: '400px', boxShadow: '0 20px 40px rgba(0,0,0,0.2)' }}
                            onClick={e => e.stopPropagation()}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                                <h3 style={{ margin: 0 }}>New Appointment</h3>
                                <button onClick={() => setIsAddingAppointment(false)} style={{ background: 'none', border: 'none', cursor: 'pointer' }}><X /></button>
                            </div>
                            <form onSubmit={handleAddAppointment}>
                                <input required placeholder="Checkup Name" style={{ width: '100%', padding: '0.8rem', borderRadius: '12px', border: '1px solid #e2e8f0', marginBottom: '0.8rem' }} value={newAppointment.title} onChange={e => setNewAppointment({ ...newAppointment, title: e.target.value })} />
                                <div style={{ display: 'flex', gap: '0.8rem', marginBottom: '0.8rem' }}>
                                    <input required type="date" style={{ flex: 1, padding: '0.8rem', borderRadius: '12px', border: '1px solid #e2e8f0' }} value={newAppointment.date} onChange={e => setNewAppointment({ ...newAppointment, date: e.target.value })} />
                                    <input required type="time" style={{ flex: 1, padding: '0.8rem', borderRadius: '12px', border: '1px solid #e2e8f0' }} value={newAppointment.time} onChange={e => setNewAppointment({ ...newAppointment, time: e.target.value })} />
                                </div>
                                <input required placeholder="Location" style={{ width: '100%', padding: '0.8rem', borderRadius: '12px', border: '1px solid #e2e8f0', marginBottom: '1rem' }} value={newAppointment.location} onChange={e => setNewAppointment({ ...newAppointment, location: e.target.value })} />
                                <button type="submit" className="btn-primary" style={{ width: '100%', padding: '0.8rem' }}>Add to Calendar</button>
                            </form>
                        </Motion.div>
                    </Motion.div>
                )}
            </AnimatePresence>
        </Motion.div>
    );
};

export default Dashboard;
