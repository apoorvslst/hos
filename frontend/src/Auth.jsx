import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Phone, Lock, User, ArrowLeft, Heart, X } from 'lucide-react';

const Auth = ({ onClose, onAuthSuccess }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [step, setStep] = useState('auth'); // 'auth' | 'profile'
    const [authMethod, setAuthMethod] = useState('email'); // 'email' or 'phone'
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        phoneNumber: '',
        password: '',
        confirmPassword: ''
    });
    const [profileData, setProfileData] = useState({
        allergies: '',
        history: '',
        pregnancyDate: ''
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
        setError('');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        const endpoint = isLogin ? '/api/auth/login' : '/api/auth/signup';
        const payload = isLogin
            ? {
                identifier: authMethod === 'email' ? formData.email : formData.phoneNumber,
                password: formData.password
            }
            : {
                name: formData.name,
                email: formData.email || undefined,
                phoneNumber: formData.phoneNumber || undefined,
                password: formData.password
            };

        try {
            // For hackathon/demo purposes, if the backend on 5000 isn't available, 
            // we'll simulate a successful login/signup for the user.
            let userData;
            try {
                const response = await fetch(`${import.meta.env.VITE_API_URL}${endpoint}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.message || 'Something went wrong');
                userData = data.user;
                if (data.token) localStorage.setItem('token', data.token);
            } catch (e) {
                console.warn("Backend auth failed, using mock data for demo:", e);
                userData = {
                    id: 'mock-id',
                    name: formData.name || 'Pragati',
                    email: formData.email,
                    phoneNumber: formData.phoneNumber
                };
            }

            if (isLogin) {
                localStorage.setItem('user', JSON.stringify(userData));
                onAuthSuccess(userData);
                onClose();
            } else {
                // If Signup, move to profile onboarding
                setFormData(prev => ({ ...prev, currentAuthUser: userData }));
                setStep('profile');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleProfileSubmit = (e) => {
        e.preventDefault();
        setLoading(true);

        const calculatePregnancyWeek = (dateStr) => {
            if (!dateStr) return "Unknown";
            const start = new Date(dateStr);
            const now = new Date();
            const diffDays = Math.floor((now - start) / (1000 * 60 * 60 * 24));
            const weeks = Math.floor(diffDays / 7);
            return weeks > 0 ? weeks : 0;
        };

        const week = calculatePregnancyWeek(profileData.pregnancyDate);
        const profileString = `Patient Case: ${week} weeks pregnant (LMP: ${profileData.pregnancyDate}). Allergies: ${profileData.allergies || 'None'}. Medical History: ${profileData.history || 'No significant history'}.`;

        const updatedUser = {
            ...formData.currentAuthUser,
            patientProfile: profileString,
            pregnancyWeek: week,
            pregnancyDate: profileData.pregnancyDate,
            allergies: profileData.allergies,
            medicalHistory: profileData.history
        };

        localStorage.setItem('user', JSON.stringify(updatedUser));
        onAuthSuccess(updatedUser);
        onClose();
        setLoading(false);
    };

    const handleProfileChange = (e) => {
        setProfileData({ ...profileData, [e.target.name]: e.target.value });
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                background: 'rgba(74, 14, 46, 0.4)',
                backdropFilter: 'blur(12px)',
                zIndex: 2000,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                padding: '1rem'
            }}
        >
            <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.9, y: 20 }}
                style={{
                    background: 'rgba(255, 255, 255, 0.95)',
                    width: '100%',
                    maxWidth: '480px',
                    borderRadius: '40px',
                    padding: '3rem',
                    boxShadow: '0 25px 50px -12px rgba(14, 165, 233, 0.25)',
                    position: 'relative',
                    overflow: 'hidden',
                    border: '1px solid rgba(255, 255, 255, 0.5)'
                }}
            >
                <button
                    onClick={onClose}
                    style={{
                        position: 'absolute',
                        top: '1.5rem',
                        right: '1.5rem',
                        background: 'rgba(14, 165, 233, 0.05)',
                        border: 'none',
                        width: '36px',
                        height: '36px',
                        borderRadius: '50%',
                        cursor: 'pointer',
                        color: 'var(--primary)',
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        zIndex: 10,
                        transition: 'all 0.3s'
                    }}
                >
                    <X size={20} />
                </button>

                <AnimatePresence mode="wait">
                    {step === 'auth' ? (
                        <motion.div
                            key="auth-step"
                            initial={{ x: 0, opacity: 1 }}
                            exit={{ x: -100, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
                                <motion.div
                                    initial={{ scale: 0.8 }}
                                    animate={{ scale: 1 }}
                                    style={{ display: 'inline-flex', alignItems: 'center', gap: '0.8rem', marginBottom: '1.5rem' }}
                                >
                                    <div style={{ background: 'var(--primary)', padding: '0.6rem', borderRadius: '15px' }}>
                                        <Heart color="white" size={32} fill="white" />
                                    </div>
                                    <span style={{ fontSize: '2.2rem', fontWeight: '800', color: 'var(--primary)', fontFamily: 'Outfit', letterSpacing: '-0.02em' }}>Fizician</span>
                                </motion.div>
                                <h2 style={{ fontSize: '1.6rem', fontWeight: '700', color: 'var(--text-dark)', marginBottom: '0.5rem' }}>
                                    {isLogin ? 'Welcome Back' : 'Get Started'}
                                </h2>
                                <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>
                                    {isLogin ? 'Sign in to access your health dashboard' : 'Join us for a healthier journey'}
                                </p>
                            </div>

                            <div style={{
                                display: 'flex',
                                background: '#f8f1f4',
                                borderRadius: '20px',
                                padding: '0.4rem',
                                marginBottom: '2.5rem',
                                position: 'relative'
                            }}>
                                <button
                                    onClick={() => setAuthMethod('email')}
                                    style={{
                                        flex: 1,
                                        padding: '0.9rem',
                                        border: 'none',
                                        borderRadius: '16px',
                                        background: authMethod === 'email' ? 'white' : 'transparent',
                                        color: authMethod === 'email' ? 'var(--primary)' : 'var(--text-light)',
                                        fontWeight: '700',
                                        cursor: 'pointer',
                                        transition: 'all 0.3s ease',
                                        boxShadow: authMethod === 'email' ? '0 4px 15px rgba(14, 165, 233, 0.1)' : 'none'
                                    }}
                                >
                                    Email
                                </button>
                                <button
                                    onClick={() => setAuthMethod('phone')}
                                    style={{
                                        flex: 1,
                                        padding: '0.9rem',
                                        border: 'none',
                                        borderRadius: '16px',
                                        background: authMethod === 'phone' ? 'white' : 'transparent',
                                        color: authMethod === 'phone' ? 'var(--primary)' : 'var(--text-light)',
                                        fontWeight: '700',
                                        cursor: 'pointer',
                                        transition: 'all 0.3s ease',
                                        boxShadow: authMethod === 'phone' ? '0 4px 15px rgba(14, 165, 233, 0.1)' : 'none'
                                    }}
                                >
                                    Phone
                                </button>
                            </div>

                            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                {!isLogin && (
                                    <div style={{ position: 'relative' }}>
                                        <User size={20} style={{ position: 'absolute', left: '1.2rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--primary)', opacity: 0.6 }} />
                                        <input
                                            type="text"
                                            name="name"
                                            placeholder="Full Name"
                                            required
                                            value={formData.name}
                                            onChange={handleChange}
                                            style={{
                                                width: '100%',
                                                padding: '1.1rem 1.1rem 1.1rem 3.5rem',
                                                borderRadius: '18px',
                                                border: '2px solid #f1f5f9',
                                                outline: 'none',
                                                fontSize: '1rem',
                                                transition: 'all 0.3s',
                                                background: '#f8fafc'
                                            }}
                                        />
                                    </div>
                                )}

                                {authMethod === 'email' ? (
                                    <div style={{ position: 'relative' }}>
                                        <Mail size={20} style={{ position: 'absolute', left: '1.2rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--primary)', opacity: 0.6 }} />
                                        <input
                                            type="email"
                                            name="email"
                                            placeholder="Email Address"
                                            required
                                            value={formData.email}
                                            onChange={handleChange}
                                            style={{
                                                width: '100%',
                                                padding: '1.1rem 1.1rem 1.1rem 3.5rem',
                                                borderRadius: '18px',
                                                border: '2px solid #f1f5f9',
                                                outline: 'none',
                                                fontSize: '1rem',
                                                background: '#f8fafc'
                                            }}
                                        />
                                    </div>
                                ) : (
                                    <div style={{ position: 'relative' }}>
                                        <Phone size={20} style={{ position: 'absolute', left: '1.2rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--primary)', opacity: 0.6 }} />
                                        <input
                                            type="tel"
                                            name="phoneNumber"
                                            placeholder="Phone Number"
                                            required
                                            value={formData.phoneNumber}
                                            onChange={handleChange}
                                            style={{
                                                width: '100%',
                                                padding: '1.1rem 1.1rem 1.1rem 3.5rem',
                                                borderRadius: '18px',
                                                border: '2px solid #f1f5f9',
                                                outline: 'none',
                                                fontSize: '1rem',
                                                background: '#f8fafc'
                                            }}
                                        />
                                    </div>
                                )}

                                <div style={{ position: 'relative' }}>
                                    <Lock size={20} style={{ position: 'absolute', left: '1.2rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--primary)', opacity: 0.6 }} />
                                    <input
                                        type="password"
                                        name="password"
                                        placeholder="Password"
                                        required
                                        value={formData.password}
                                        onChange={handleChange}
                                        style={{
                                            width: '100%',
                                            padding: '1.1rem 1.1rem 1.1rem 3.5rem',
                                            borderRadius: '18px',
                                            border: '2px solid #f1f5f9',
                                            outline: 'none',
                                            fontSize: '1rem',
                                            background: '#f8fafc'
                                        }}
                                    />
                                </div>

                                {error && (
                                    <motion.p
                                        initial={{ opacity: 0 }}
                                        animate={{ opacity: 1 }}
                                        style={{ color: '#e74c3c', fontSize: '0.9rem', textAlign: 'center', fontWeight: '600' }}
                                    >
                                        {error}
                                    </motion.p>
                                )}

                                <button
                                    type="submit"
                                    disabled={loading}
                                    style={{
                                        padding: '1.1rem',
                                        fontSize: '1.1rem',
                                        marginTop: '0.5rem',
                                        borderRadius: '18px',
                                        background: 'var(--primary)',
                                        color: 'white',
                                        border: 'none',
                                        fontWeight: '700',
                                        cursor: 'pointer',
                                        boxShadow: '0 10px 20px rgba(14, 165, 233, 0.2)',
                                        transition: 'all 0.3s'
                                    }}
                                >
                                    {loading ? 'Thinking...' : (isLogin ? 'Sign In' : 'Create Account')}
                                </button>
                            </form>

                            <div style={{ textAlign: 'center', marginTop: '2.5rem' }}>
                                <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>
                                    {isLogin ? "Don't have an account?" : "Already have an account?"}
                                    <button
                                        onClick={() => setIsLogin(!isLogin)}
                                        style={{
                                            background: 'none',
                                            border: 'none',
                                            color: 'var(--primary)',
                                            fontWeight: '800',
                                            marginLeft: '0.5rem',
                                            cursor: 'pointer',
                                            textDecoration: 'underline'
                                        }}
                                    >
                                        {isLogin ? 'Sign Up' : 'Log In'}
                                    </button>
                                </p>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="profile-step"
                            initial={{ x: 100, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            transition={{ duration: 0.3 }}
                        >
                            <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
                                <div style={{
                                    width: '80px',
                                    height: '80px',
                                    background: '#fff0f5',
                                    borderRadius: '50%',
                                    display: 'flex',
                                    justifyContent: 'center',
                                    alignItems: 'center',
                                    margin: '0 auto 1.5rem',
                                    color: 'var(--primary)'
                                }}>
                                    <User size={40} />
                                </div>
                                <h2 style={{ fontSize: '1.6rem', fontWeight: '700', color: 'var(--text-dark)', marginBottom: '0.5rem' }}>
                                    Tell us about yourself
                                </h2>
                                <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>
                                    This helps Fizician provide accurate medical advice.
                                </p>
                            </div>

                            <form onSubmit={handleProfileSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    <label style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-dark)', marginLeft: '0.5rem' }}>Pregnancy Start Date (LMP)</label>
                                    <input
                                        type="date"
                                        name="pregnancyDate"
                                        required
                                        value={profileData.pregnancyDate}
                                        onChange={handleProfileChange}
                                        style={{
                                            width: '100%',
                                            padding: '1.1rem',
                                            borderRadius: '18px',
                                            border: '2px solid #f1f5f9',
                                            outline: 'none',
                                            fontSize: '1rem',
                                            background: '#f8fafc'
                                        }}
                                    />
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    <label style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-dark)', marginLeft: '0.5rem' }}>Any Allergies?</label>
                                    <textarea
                                        name="allergies"
                                        placeholder="e.g. Peanuts, Penicillin, etc."
                                        value={profileData.allergies}
                                        onChange={handleProfileChange}
                                        style={{
                                            width: '100%',
                                            padding: '1.1rem',
                                            borderRadius: '18px',
                                            border: '2px solid #f1f5f9',
                                            outline: 'none',
                                            fontSize: '1rem',
                                            background: '#f8fafc',
                                            minHeight: '80px',
                                            resize: 'none',
                                            fontFamily: 'inherit'
                                        }}
                                    />
                                </div>

                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    <label style={{ fontSize: '0.9rem', fontWeight: '700', color: 'var(--text-dark)', marginLeft: '0.5rem' }}>Medical History</label>
                                    <textarea
                                        name="history"
                                        placeholder="Thyroid, BP issues, previous surgery, etc."
                                        value={profileData.history}
                                        onChange={handleProfileChange}
                                        style={{
                                            width: '100%',
                                            padding: '1.1rem',
                                            borderRadius: '18px',
                                            border: '2px solid #f1f5f9',
                                            outline: 'none',
                                            fontSize: '1rem',
                                            background: '#f8fafc',
                                            minHeight: '80px',
                                            resize: 'none',
                                            fontFamily: 'inherit'
                                        }}
                                    />
                                </div>

                                <button
                                    type="submit"
                                    disabled={loading}
                                    style={{
                                        padding: '1.1rem',
                                        fontSize: '1.1rem',
                                        marginTop: '1rem',
                                        borderRadius: '18px',
                                        background: 'var(--primary)',
                                        color: 'white',
                                        border: 'none',
                                        fontWeight: '700',
                                        cursor: 'pointer',
                                        boxShadow: '0 10px 20px rgba(14, 165, 233, 0.2)',
                                        transition: 'all 0.3s'
                                    }}
                                >
                                    {loading ? 'Saving Profile...' : 'Complete Setup'}
                                </button>
                            </form>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        </motion.div>
    );
};

export default Auth;
