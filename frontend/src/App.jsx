import React, { useState, useEffect } from 'react';
import { motion as Motion, AnimatePresence } from 'framer-motion';
import { Link as ScrollLink, Element } from 'react-scroll';
import {
  Mic,
  Heart,
  Baby,
  BookOpen,
  Users,
  ArrowRight,
  Play,
  ShieldCheck,
  Activity,
  MessageCircle,
  Stethoscope,
  LogOut,
  User as UserIcon,
  MessageSquare,
  Volume2,
  Bot,
  Calendar,
  Award,
  TrendingUp,
  AlertTriangle,
  ChevronRight,
  Clock,
  Utensils,
  Globe,
  Loader,
  Ear,
  RotateCcw,
  Phone,
  X
} from 'lucide-react';
import Auth from './Auth';
import Dashboard from './Dashboard';


// Hero Image Path from the user link
const HERO_IMAGE = 'https://clipart-library.com/2024/pregnant-woman-cartoon/pregnant-woman-cartoon-1.jpg';

const Navbar = ({ onAuthClick, user, onLogout, onContactClick, contactLoading, setView, currentView }) => {
  return (
    <nav className="glass-nav">
      <div className="container py-4 flex justify-between items-center" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
        <ScrollLink
          to="home"
          smooth={true}
          duration={500}
          style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          <span style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--primary)', fontFamily: 'Outfit' }}>Janani</span>
        </ScrollLink>
        <div className="nav-links" style={{ display: 'flex', gap: '2rem' }}>
          {['Home', 'Demo', 'Mission'].map((item) => (
            <ScrollLink
              key={item}
              to={item.toLowerCase()}
              smooth={true}
              duration={500}
              offset={-70}
              onClick={() => setView('landing')}
              style={{
                cursor: 'pointer',
                fontWeight: '500',
                color: currentView === 'landing' ? 'var(--text-dark)' : '#64748b',
                transition: 'color 0.3s'
              }}
              className="hover-text-primary"
            >
              {item}
            </ScrollLink>
          ))}
          {user && (
            <button
              onClick={() => setView('dashboard')}
              style={{
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                fontWeight: '500',
                color: currentView === 'dashboard' ? 'var(--primary)' : '#64748b',
                fontFamily: 'inherit',
                fontSize: 'inherit',
                transition: 'color 0.3s'
              }}
              className="hover-text-primary"
            >
              Dashboard
            </button>
          )}
        </div>
        <div className="nav-actions" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <button
            onClick={onContactClick}
            disabled={contactLoading}
            className="btn-primary nav-contact-btn"
            style={{
              padding: '0.5rem 1.2rem',
              background: 'transparent',
              border: '2px solid var(--primary)',
              color: 'var(--primary)',
              opacity: contactLoading ? 0.7 : 1
            }}
          >
            {contactLoading ? 'Calling...' : 'Contact'}
          </button>
          {user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div className="nav-user-name" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)', fontWeight: '600' }}>
                <UserIcon size={18} />
                <span>{user.name}</span>
              </div>
              <button onClick={onLogout} className="btn-primary" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <LogOut size={16} /> Logout
              </button>
            </div>
          ) : (
            <button onClick={onAuthClick} className="btn-primary" style={{ padding: '0.5rem 1.2rem' }}>Login</button>
          )}
        </div>
      </div>
    </nav>
  );
};


const Hero = ({ onAuthClick, user, setView }) => {
  const [petals, setPetals] = useState([]);

  const spawnPetals = (x, y) => {
    const newPetals = Array.from({ length: 12 }).map(() => ({
      id: Math.random(),
      x,
      y,
      size: Math.random() * 15 + 10,
      rotation: Math.random() * 360,
      drift: (Math.random() - 0.5) * 200,
      duration: Math.random() * 2 + 2,
    }));
    setPetals((prev) => [...prev, ...newPetals]);
  };

  return (
    <Element name="home" onClick={(e) => spawnPetals(e.clientX, e.clientY)}>
      <div style={{ position: 'fixed', top: 0, left: 0, pointerEvents: 'none', zIndex: 9999 }}>
        <AnimatePresence>
          {petals.map((petal) => (
            <Motion.div
              key={petal.id}
              initial={{ opacity: 0.8, scale: 0, x: petal.x, y: petal.y, rotate: petal.rotation }}
              animate={{
                opacity: 0,
                scale: [0.5, 1, 0.8],
                y: petal.y + 400 + Math.random() * 200,
                x: petal.x + petal.drift,
                rotate: petal.rotation + (Math.random() > 0.5 ? 180 : -180)
              }}
              transition={{ duration: petal.duration, ease: "easeOut" }}
              onAnimationComplete={() => setPetals((prev) => prev.filter((p) => p.id !== petal.id))}
              style={{
                position: 'absolute',
                width: petal.size,
                height: petal.size * 0.7,
                background: 'linear-gradient(135deg, #ff69b4, #fce4ec)',
                borderRadius: '50% 0 50% 50%',
                boxShadow: '0 2px 5px rgba(255, 105, 180, 0.2)',
              }}
            />
          ))}
        </AnimatePresence>
      </div>
      <section className="container" style={{ paddingTop: '100px', cursor: 'pointer' }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4rem',
          flexDirection: 'row',
          flexWrap: 'wrap'
        }}>
          {/* Left Content */}
          <Motion.div
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8 }}
            style={{ flex: '1', minWidth: '300px', textAlign: 'left' }}
          >
            <Motion.h1
              className="gradient-text"
              style={{ fontSize: '5.5rem', lineHeight: '1.1', marginBottom: '1.5rem', fontWeight: '800' }}
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              Janani
            </Motion.h1>
            <h2 style={{ fontSize: '2.2rem', color: 'var(--text-dark)', marginBottom: '2rem', fontWeight: '600', lineHeight: '1.3' }}>
              AI-Powered Multilingual Voice Assistant for <span style={{ color: 'var(--primary)' }}>Rural Maternal Care</span>
            </h2>

            <p style={{ fontSize: '1.25rem', color: 'var(--text-light)', marginBottom: '3rem', maxWidth: '550px' }}>
              Empowering Rural Women with RAG-Based Medical Intelligence.
              Providing life-saving prenatal care insights through simple, natural voice conversations.
            </p>

            <div style={{ display: 'flex', gap: '1.5rem' }}>
              {!user && (
                <button
                  onClick={onAuthClick}
                  className="btn-primary"
                  style={{ padding: '1rem 2.5rem', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  Get Started <ArrowRight size={20} />
                </button>
              )}
              {user && (
                <button
                  onClick={() => setView('dashboard')}
                  className="btn-primary"
                  style={{ padding: '1rem 2.5rem', fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                >
                  Go to Dashboard <ArrowRight size={20} />
                </button>
              )}
              <ScrollLink to="mission" smooth={true} duration={800} offset={-70}>
                <button style={{
                  background: 'transparent',
                  border: '2px solid var(--primary)',
                  color: 'var(--primary)',
                  padding: '1rem 2rem',
                  borderRadius: '50px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}>
                  Learn More
                </button>
              </ScrollLink>
            </div>
          </Motion.div>


          {/* Right Image */}
          <Motion.div
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            style={{ flex: '1', minWidth: '300px', display: 'flex', justifyContent: 'center', position: 'relative' }}
          >
            <div style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: '115%',
              height: '115%',
              background: 'var(--accent)',
              borderRadius: '50%',
              filter: 'blur(60px)',
              zIndex: -1,
              opacity: 0.6
            }}></div>
            <img
              src={HERO_IMAGE}
              alt="Maternal Health"
              style={{
                width: '100%',
                maxWidth: '450px',
                height: '450px',
                objectFit: 'cover',
                borderRadius: '50%',
              }}
              onError={(e) => {
                e.target.src = 'https://images.unsplash.com/photo-1559832306-27a0278d99a7?auto=format&fit=crop&q=80&w=1000';
              }}
            />
          </Motion.div>
        </div>
      </section>
    </Element>
  );
};

const DemoSection = () => {
  return (
    <Element name="demo">
      <section style={{ background: '#fff' }}>
        <div className="container text-center" style={{ textAlign: 'center' }}>
          <Motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 style={{ fontSize: '3rem', marginBottom: '1rem' }}>Seeing is Believing</h2>
            <p style={{ maxWidth: '700px', margin: '0 auto 4rem', color: 'var(--text-light)' }}>
              Watch how MatriCare AI transforms the complex prenatal tracking process into a simple, natural conversation.
            </p>

            <div style={{
              width: '100%',
              maxWidth: '900px',
              margin: '0 auto',
              aspectRatio: '16/9',
              background: '#000',
              borderRadius: '30px',
              overflow: 'hidden',
              position: 'relative',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
            }}>
              {/* Mock Video Player */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                background: 'linear-gradient(45deg, rgba(0,0,0,0.7), rgba(0,0,0,0.3))'
              }}>
                <Motion.div
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  style={{
                    cursor: 'pointer',
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    background: 'white',
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    marginBottom: '1.5rem'
                  }}
                >
                  <Play fill="black" size={32} style={{ marginLeft: '5px' }} />
                </Motion.div>
                <div style={{ color: 'white', fontSize: '1.2rem', fontWeight: '500' }}>Demo Video: Voice-Based Tracking</div>
              </div>
              <img
                src="https://images.unsplash.com/photo-1584362946444-1e7c4f440e91?auto=format&fit=crop&q=80&w=1000"
                style={{ width: '100%', height: '100%', objectFit: 'cover', opacity: 0.6 }}
                alt="Demo Preview"
              />
            </div>
          </Motion.div>
        </div>
      </section>
    </Element>
  );
};

const FeatureCard = ({ icon: Icon, title, desc, delay }) => (
  <Motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay, duration: 0.6 }}
    className="hover-lift"
    style={{
      background: 'white',
      padding: '2.5rem',
      borderRadius: '24px',
      textAlign: 'left',
      boxShadow: 'var(--shadow-md)',
      border: '1px solid #f0f0f0'
    }}
  >
    <div style={{
      background: '#f1f8e9',
      width: '60px',
      height: '60px',
      borderRadius: '16px',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      marginBottom: '1.5rem'
    }}>
      <Icon color="var(--primary)" size={32} />
    </div>
    <h3 style={{ marginBottom: '1rem', fontSize: '1.5rem' }}>{title}</h3>
    <p style={{ color: 'var(--text-light)', fontSize: '0.95rem' }}>{desc}</p>
  </Motion.div>
);

const SARVAM_API_KEY = "sk_94vvqhgo_opzIH8VOZKtoPs894jfnFGAZ";

const SUPPORTED_LANGUAGES = [
  { code: 'unknown', label: 'Auto Detect', native: 'स्वचालित पहचान' },
  { code: 'hi-IN', label: 'Hindi', native: 'हिंदी' },
  { code: 'en-IN', label: 'English', native: 'English' },
  { code: 'pa-IN', label: 'Punjabi', native: 'ਪੰਜਾਬੀ' },
  { code: 'bn-IN', label: 'Bengali', native: 'বাংলা' },
  { code: 'mr-IN', label: 'Marathi', native: 'मराठी' },
  { code: 'gu-IN', label: 'Gujarati', native: 'ગુજરાતી' },
  { code: 'ta-IN', label: 'Tamil', native: 'தமிழ்' },
  { code: 'te-IN', label: 'Telugu', native: 'తెలుగు' },
  { code: 'kn-IN', label: 'Kannada', native: 'ಕನ್ನಡ' },
  { code: 'ml-IN', label: 'Malayalam', native: 'മലയാളം' },
  { code: 'or-IN', label: 'Odia', native: 'ଓଡ଼ିଆ' },
  { code: 'as-IN', label: 'Assamese', native: 'অসমীয়া' },
  { code: 'ur-IN', label: 'Urdu', native: 'اردو' },
  { code: 'sa-IN', label: 'Sanskrit', native: 'संस्कृतम्' },
  { code: 'ks-IN', label: 'Kashmiri', native: 'کأشُر' },
  { code: 'sd-IN', label: 'Sindhi', native: 'سنڌي' },
  { code: 'doi-IN', label: 'Dogri', native: 'डोगरी' },
  { code: 'kok-IN', label: 'Konkani', native: 'कोंकणी' },
  { code: 'mni-IN', label: 'Manipuri', native: 'মণিপুরী' },
  { code: 'mai-IN', label: 'Maithili', native: 'मैथिली' },
  { code: 'sat-IN', label: 'Santali', native: 'संताली' },
  { code: 'ne-NP', label: 'Nepali', native: 'नेपाली' }
];

const VoiceInterface = ({ user }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [showTextInput, setShowTextInput] = useState(false);
  const [textInput, setTextInput] = useState('');
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [interactionMode, setInteractionMode] = useState('IDLE'); // IDLE, LISTENING, PROCESSING, SPEAKING
  const [activeInputMode, setActiveInputMode] = useState('voice'); // 'text' | 'voice'
  const [selectedLanguage, setSelectedLanguage] = useState('unknown');
  const [detectedLang, setDetectedLang] = useState(null);
  const [lastAudioBase64, setLastAudioBase64] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioChunksRef = React.useRef([]);
  const audioPlayerRef = React.useRef(new Audio());
  const holdTimerRef = React.useRef(null);
  const isHoldingRef = React.useRef(false);

  useEffect(() => {
    audioPlayerRef.current.onended = () => {
      setInteractionMode('IDLE');
    };
  }, []);

  const resetSession = () => {
    audioPlayerRef.current.pause();
    audioPlayerRef.current.currentTime = 0;
    setInteractionMode('IDLE');
    setTranscript('');
    setResponse('');
    setDetectedLang(null);
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
  };

  const handleOrbClick = () => {
    if (interactionMode === 'IDLE' || interactionMode === 'ERROR') {
      startRecording();
    } else if (interactionMode === 'LISTENING') {
      stopRecording();
    } else {
      // If Speaking or Processing, a click resets the session
      resetSession();
    }
  };

  const handleRepeatAudio = () => {
    if (lastAudioBase64) {
      audioPlayerRef.current.currentTime = 0;
      setInteractionMode('SPEAKING');
      audioPlayerRef.current.play();
    }
  };

  const handleSendText = async () => {
    if (!textInput.trim()) return;

    setIsLoading(true);
    setResponse('');
    setTranscript(`You: "${textInput}"`);
    const queryTerm = textInput;
    setTextInput('');

    try {
      setInteractionMode('PROCESSING');
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryTerm,
          language_code: selectedLanguage === 'unknown' ? 'en-IN' : selectedLanguage,
          patient_data: user?.patientProfile || "General pregnancy wellness query from text input.",
          user_phone: user?.phoneNumber || '',
          user_email: user?.email || '',
          user_name: user?.name || '',
          source: 'website'
        })
      });

      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data = await res.json();
      if (data.status === 'success') {
        const localizedAnswer = data.localized_answer || data.english_answer || "I'm sorry, I couldn't generate a specific answer.";
        setResponse(localizedAnswer);

        // TTS via Sarvam
        if (localizedAnswer.trim()) {
          await speakText(localizedAnswer, selectedLanguage === 'unknown' ? 'hi-IN' : selectedLanguage);
        }
      } else {
        setResponse("I'm sorry, I encountered an error while processing your request.");
        setInteractionMode('IDLE');
      }
    } catch (error) {
      console.error("Error calling RAG service:", error);
      setResponse("Could not connect to the Janani AI service.");
      setInteractionMode('IDLE');
    } finally {
      setIsLoading(false);
    }
  };

  const speakText = async (text, langCode) => {
    if (!text || !text.trim()) {
      console.warn("TTS skipped: empty text");
      return;
    }

    try {
      setInteractionMode('PROCESSING');

      // Map full language names or codes to Sarvam v3 -IN format
      let targetLang = langCode;
      const langLower = targetLang?.toLowerCase() || 'hindi';

      const langMap = {
        'marathi': 'mr-IN',
        'hindi': 'hi-IN',
        'bengali': 'bn-IN',
        'telugu': 'te-IN',
        'tamil': 'ta-IN',
        'gujarati': 'gu-IN',
        'kannada': 'kn-IN',
        'malayalam': 'ml-IN',
        'punjabi': 'pa-IN',
        'assamese': 'as-IN',
        'odia': 'or-IN',
        'sanskrit': 'sa-IN',
        'urdu': 'ur-IN',
        'mr': 'mr-IN',
        'hi': 'hi-IN',
        'bn': 'bn-IN',
        'te': 'te-IN',
        'ta': 'ta-IN',
        'gu': 'gu-IN',
        'kn': 'kn-IN',
        'ml': 'ml-IN',
        'pa': 'pa-IN',
        'as': 'as-IN',
        'or': 'or-IN',
        'ur': 'ur-IN'
      };

      if (langMap[langLower]) {
        targetLang = langMap[langLower];
      } else if (targetLang && !targetLang.includes('-')) {
        targetLang = `${targetLang}-IN`;
      }

      if (targetLang?.toLowerCase().includes('unknown')) targetLang = 'hi-IN';
      if (targetLang?.toLowerCase().includes('english')) targetLang = 'hi-IN'; // Fallback to Hindi speech if English encountered

      // ENSURE TEXT IS WITHIN SARVAM LIMIT (500 CHARS)
      const cleanText = text.length > 500 ? text.substring(0, 497) + "..." : text;

      const response = await fetch('https://api.sarvam.ai/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-subscription-key': SARVAM_API_KEY
        },
        body: JSON.stringify({
          inputs: [cleanText],
          target_language_code: targetLang,
          speaker: "priya",
          model: "bulbul:v3"
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Sarvam TTS error: ${response.status} - ${JSON.stringify(errorData)}`);
      }

      const data = await response.json();
      if (!data.audios || !data.audios[0]) {
        throw new Error("Sarvam TTS returned success but no audio data.");
      }

      const audioBase64 = data.audios[0];
      setLastAudioBase64(audioBase64);
      audioPlayerRef.current.src = `data:audio/wav;base64,${audioBase64}`;
      setInteractionMode('SPEAKING');
      await audioPlayerRef.current.play();
    } catch (err) {
      console.error("TTS failed:", err);
      setInteractionMode('IDLE');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 16000 }
      });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      audioChunksRef.current = [];
      recorder.ondataavailable = e => { if (e.data.size > 0) audioChunksRef.current.push(e.data); };
      recorder.onstop = processAudio;
      recorder.start();
      setMediaRecorder(recorder);
      setIsRecording(true);
      setInteractionMode('LISTENING');
      setTranscript('Listening carefully...');
      setResponse('');
    } catch (err) {
      console.error("Mic access denied:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    setIsRecording(false);
  };

  const processAudio = async () => {
    setInteractionMode('PROCESSING');
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    if (audioBlob.size < 1000) {
      setInteractionMode('IDLE');
      setTranscript('Audio too short.');
      return;
    }

    try {
      // 1. STT via Sarvam
      setTranscript('Transcribing...');
      const sttResult = await callSarvamSTT(audioBlob);
      const userText = sttResult.transcript;
      const langCode = sttResult.detectedLangCode;
      setDetectedLang(langCode);
      setTranscript(`You: "${userText}"`);

      // 2. RAG via Backend
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userText,
          language_code: langCode,
          patient_data: user?.patientProfile || "Voice query from rural patient.",
          user_phone: user?.phoneNumber || '',
          user_email: user?.email || '',
          user_name: user?.name || '',
          source: 'website'
        })
      });
      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data = await res.json();

      // Sync verified language from backend
      const finalLang = data.verified_language || langCode;
      setDetectedLang(finalLang);

      const localizedAnswer = data.localized_answer || data.english_answer || "Diagnosis complete.";
      setResponse(localizedAnswer);

      // 3. TTS via Sarvam
      if (localizedAnswer.trim()) {
        await speakText(localizedAnswer, finalLang);
      }

    } catch (error) {
      console.error("Voice processing failed:", error);
      setInteractionMode('IDLE');
    }
  };

  const callSarvamSTT = async (audioBlob) => {
    const formData = new FormData();
    formData.append('file', audioBlob, 'recording.webm');
    formData.append('model', 'saaras:v3');
    // No fallback hint; Sarvam saaras:v3 handles auto-detection effectively.
    if (selectedLanguage !== 'unknown') {
      formData.append('language_code', selectedLanguage);
    }

    const res = await fetch('https://api.sarvam.ai/speech-to-text', {
      method: 'POST',
      headers: { 'api-subscription-key': SARVAM_API_KEY },
      body: formData
    });
    const data = await res.json();
    return { transcript: data.transcript, detectedLangCode: data.language_code || (selectedLanguage === 'unknown' ? 'hi-IN' : selectedLanguage) };
  };

  const handlePointerDown = (e) => {
    e.preventDefault();
    isHoldingRef.current = false;
    holdTimerRef.current = setTimeout(() => {
      isHoldingRef.current = true;
      startRecording();
    }, 400); // 400ms threshold for hold
  };

  const handlePointerUp = (e) => {
    e.preventDefault();
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      if (isHoldingRef.current) {
        stopRecording();
      } else {
        // Quick click -> Toggle Text Option
        setShowTextInput(!showTextInput);
        if (!showTextInput) {
          setTranscript('Text Mode: Typing enabled.');
        } else {
          setTranscript('');
        }
      }
      holdTimerRef.current = null;
    }
  };

  return (
    <Motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      style={{
        marginTop: '4rem',
        padding: '3rem',
        background: 'linear-gradient(135deg, #fff 0%, #fef7f9 100%)',
        borderRadius: '30px',
        boxShadow: 'var(--shadow-lg)',
        border: '1px solid var(--accent)',
        maxWidth: '800px',
        margin: '4rem auto 0',
        textAlign: 'center'
      }}
    >
      <p style={{ color: 'var(--text-light)', marginBottom: '1rem' }}>
        How would you like to interact with Janani?
      </p>

      {/* Mode Switcher */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
        <div style={{
          background: '#f1f5f9',
          padding: '4px',
          borderRadius: '50px',
          display: 'flex',
          gap: '4px'
        }}>
          <button
            onClick={() => { setActiveInputMode('voice'); resetSession(); }}
            style={{
              padding: '0.6rem 1.5rem',
              borderRadius: '50px',
              border: 'none',
              background: activeInputMode === 'voice' ? 'white' : 'transparent',
              boxShadow: activeInputMode === 'voice' ? '0 2px 8px rgba(0,0,0,0.1)' : 'none',
              color: activeInputMode === 'voice' ? 'var(--primary)' : '#64748b',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.3s'
            }}
          >
            <Mic size={18} /> Voice
          </button>
          <button
            onClick={() => { setActiveInputMode('text'); resetSession(); }}
            style={{
              padding: '0.6rem 1.5rem',
              borderRadius: '50px',
              border: 'none',
              background: activeInputMode === 'text' ? 'white' : 'transparent',
              boxShadow: activeInputMode === 'text' ? '0 2px 8px rgba(0,0,0,0.1)' : 'none',
              color: activeInputMode === 'text' ? 'var(--primary)' : '#64748b',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              transition: 'all 0.3s'
            }}
          >
            <MessageSquare size={18} /> Text
          </button>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2.5rem' }}>
        <div style={{
          background: 'white',
          padding: '0.75rem 1.25rem',
          borderRadius: '20px',
          border: '1px solid var(--accent)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.8rem',
          boxShadow: '0 4px 12px rgba(176, 24, 84, 0.05)',
          width: '100%',
          maxWidth: '350px'
        }}>
          <Globe size={20} color="var(--primary)" />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
            <span style={{ fontSize: '0.7rem', fontWeight: 'bold', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Language (भाषा)</span>
            <select
              value={selectedLanguage}
              onChange={(e) => setSelectedLanguage(e.target.value)}
              style={{
                border: 'none',
                outline: 'none',
                background: 'transparent',
                fontWeight: '700',
                fontSize: '1.05rem',
                color: 'var(--text-dark)',
                cursor: 'pointer',
                width: '100%',
                marginTop: '2px'
              }}
            >
              {SUPPORTED_LANGUAGES.map(lang => (
                <option key={lang.code} value={lang.code}>
                  {lang.native} ({lang.label})
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div style={{ minHeight: '150px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        {activeInputMode === 'voice' ? (
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <AnimatePresence>
              {interactionMode === 'SPEAKING' && (
                [0, 1, 2].map((i) => (
                  <Motion.div
                    key={i}
                    initial={{ scale: 1, opacity: 0.6 }}
                    animate={{ scale: 2.2, opacity: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.6,
                      ease: "easeOut"
                    }}
                    style={{
                      position: 'absolute',
                      top: 0, left: 0, right: 0, bottom: 0,
                      background: interactionMode === 'LISTENING' ? '#ef4444' : (interactionMode === 'SPEAKING' ? '#3b82f6' : '#f59e0b'),
                      borderRadius: '50%',
                      zIndex: 0
                    }}
                  />
                ))
              )}
            </AnimatePresence>

            <Motion.button
              onClick={handleOrbClick}
              whileTap={{ scale: 0.92 }}
              style={{
                width: '120px',
                height: '120px',
                minWidth: '120px',
                minHeight: '120px',
                borderRadius: '50%',
                background: interactionMode === 'LISTENING' ? '#ef4444' : (interactionMode === 'SPEAKING' ? '#3b82f6' : (interactionMode === 'PROCESSING' ? '#f59e0b' : 'var(--primary)')),
                border: 'none',
                color: 'white',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                cursor: 'pointer',
                position: 'relative',
                zIndex: 10,
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.1)',
                margin: '0 auto',
                touchAction: 'none'
              }}
            >
              {interactionMode === 'LISTENING' ? (
                <Ear size={50} />
              ) : (interactionMode === 'SPEAKING' ? (
                <Motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                >
                  <Volume2 size={40} />
                </Motion.div>
              ) : (interactionMode === 'PROCESSING' ? (
                <Motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                  style={{ display: 'flex' }}
                >
                  <Loader size={40} />
                </Motion.div>
              ) : <Mic size={45} />))}
            </Motion.button>
            <p style={{ marginTop: '1rem', fontWeight: 'bold', color: 'var(--text-light)' }}>
              {interactionMode === 'IDLE' ? 'Tap to Speak' : ''}
              {interactionMode === 'LISTENING' ? 'Listening... Tap to Stop' : ''}
              {interactionMode === 'SPEAKING' ? 'Janani Speaking...' : ''}
              {interactionMode === 'PROCESSING' ? 'Thinking...' : ''}
            </p>
          </div>
        ) : (
          <div style={{ position: 'relative', maxWidth: '500px', width: '100%', margin: '0 auto' }}>
            <input
              type="text"
              placeholder="Type your health concern here..."
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendText()}
              disabled={isLoading}
              style={{
                width: '100%',
                padding: '1.2rem 1.5rem',
                borderRadius: '50px',
                border: '2px solid var(--accent)',
                fontSize: '1rem',
                outline: 'none',
                boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.02)',
                opacity: isLoading ? 0.7 : 1
              }}
            />
            <button
              onClick={handleSendText}
              disabled={isLoading}
              style={{
                position: 'absolute',
                right: '8px',
                top: '50%',
                transform: 'translateY(-50%)',
                background: 'var(--primary)',
                color: 'white',
                border: 'none',
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                cursor: 'pointer',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                opacity: isLoading ? 0.7 : 1
              }}
            >
              {isLoading ? (
                <Loader size={20} className="animate-spin" />
              ) : (
                <ArrowRight size={20} />
              )}
            </button>
          </div>
        )}
      </div>

      <div style={{ marginTop: '2rem', minHeight: '60px', width: '100%' }}>
        {transcript && (
          <Motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              fontStyle: 'italic',
              color: 'var(--primary)',
              fontWeight: '500',
              padding: '1rem',
              background: 'white',
              borderRadius: '12px',
              borderLeft: '4px solid var(--primary)',
              display: 'inline-block',
              maxWidth: '90%'
            }}
          >
            {transcript}
          </Motion.p>
        )}

        {detectedLang && (
          <div style={{ marginTop: '0.5rem' }}>
            <span style={{
              fontSize: '0.75rem',
              background: 'var(--accent)',
              color: 'var(--primary)',
              padding: '0.2rem 0.6rem',
              borderRadius: '20px',
              fontWeight: '600',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '4px'
            }}>
              <Globe size={12} />
              Language: {SUPPORTED_LANGUAGES.find(l => l.code === detectedLang || l.code.split('-')[0] === detectedLang.split('-')[0])?.label || detectedLang}
            </span>
          </div>
        )}
        {response && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              style={{
                marginTop: '1.5rem',
                padding: '1.5rem',
                background: 'white',
                borderRadius: '24px',
                border: '1px solid var(--accent)',
                boxShadow: 'var(--shadow-md)',
                textAlign: 'left',
                position: 'relative',
                width: '100%'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '1rem' }}>
                <Bot size={24} color="var(--primary)" />
                <strong style={{ color: 'var(--primary)' }}>Janani AI:</strong>
              </div>
              <p style={{ color: 'var(--text-dark)', lineHeight: '1.6', fontSize: '1.05rem' }}>{response}</p>
            </Motion.div>

            {lastAudioBase64 && (
              <button
                onClick={handleRepeatAudio}
                style={{
                  marginTop: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  padding: '0.6rem 1.2rem',
                  borderRadius: '50px',
                  background: 'white',
                  border: '1px solid var(--accent)',
                  color: 'var(--primary)',
                  fontWeight: '600',
                  cursor: 'pointer',
                  boxShadow: 'var(--shadow-sm)',
                  transition: 'background 0.3s'
                }}
              >
                <RotateCcw size={18} /> फिर से सुनें (Repeat)
              </button>
            )}
          </div>
        )}
      </div>
    </Motion.div>
  );
};

const MainContent = ({ assistantRef, user }) => {
  return (
    <section style={{ background: 'transparent', minHeight: 'auto', paddingBottom: '150px' }}>
      <div className="container">
        {/* Voice Input Interface Relocated to Top */}
        <Element name="voice-assistant">
          <div ref={assistantRef}>
            <VoiceInterface user={user} />
          </div>
        </Element>

        <Element name="mission" style={{ marginTop: '5rem', paddingTop: '5rem' }}>
          <div style={{ textAlign: 'center', marginBottom: '5rem' }}>
            <h2 style={{ fontSize: '3rem', marginBottom: '1rem' }}>The Knowledge Foundation</h2>
            <p style={{ maxWidth: '800px', margin: '0 auto', color: 'var(--text-light)' }}>
              We use Retreival Augmented Generation (RAG) to process vast amounts of medical expertise,
              ensuring every mother gets scientifically accurate, simplified advice.
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
            gap: '2.5rem'
          }}>
            <FeatureCard
              icon={BookOpen}
              title="Expert Knowledge Base"
              desc="Fed with data from MBBS, MS, MD textbooks and pregnancy research papers for high reliability."
              delay={0.1}
            />
            <FeatureCard
              icon={Stethoscope}
              title="Risk Analysis"
              desc="Scraping specialist articles to detect high-risk symptoms and unavoidable physical circumstances."
              delay={0.2}
            />
            <FeatureCard
              icon={Activity}
              title="Daily Vital Tracking"
              desc="Voice-logged records of fetal movement, symptoms, and nutrition for holistic health monitoring."
              delay={0.3}
            />
            <FeatureCard
              icon={MessageCircle}
              title="Bhavini & Google TTS"
              desc="Leveraging Indian-language APIs for local accessibility in Hindi, Marathi, and Tamil."
              delay={0.4}
            />
            <FeatureCard
              icon={ShieldCheck}
              title="Secure RAG Pipeline"
              desc="Real-time fetching and synthesis of medical data tailored to individual patient history."
              delay={0.5}
            />
            <FeatureCard
              icon={Users}
              title="Low-Literate Friendly"
              desc="Eliminating manual text input; pure voice-powered interaction for rural accessibility."
              delay={0.6}
            />
          </div>

        </Element>
      </div>
    </section>
  );
};

// ─── Contact/Support Modal with RAG Chat ──────────────────────────────────────
const ContactModal = ({ onClose, onProceed, loading, user, selectedLanguage }) => {
  const [supportInput, setSupportInput] = useState('');
  const [supportResponse, setSupportResponse] = useState('');
  const [isSupportLoading, setIsSupportLoading] = useState(false);

  const handleSupportSend = async () => {
    if (!supportInput.trim()) return;
    setIsSupportLoading(true);
    setSupportResponse('');
    const queryTerm = supportInput;
    setSupportInput('');

    try {
      const res = await fetch(`${import.meta.env.VITE_API_URL}/api/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryTerm,
          language_code: selectedLanguage === 'unknown' ? 'en-IN' : selectedLanguage,
          patient_data: user?.patientProfile || "General support query from website contact modal.",
          user_phone: user?.phoneNumber || '',
          user_email: user?.email || '',
          user_name: user?.name || '',
          source: 'support_chat'
        })
      });

      if (!res.ok) throw new Error(`Backend error: ${res.status}`);
      const data = await res.json();
      if (data.status === 'success') {
        const localizedAnswer = data.localized_answer || data.english_answer;
        setSupportResponse(localizedAnswer);
      } else {
        setSupportResponse("I'm sorry, I encountered an error while processing your request.");
      }
    } catch (error) {
      console.error("Error calling RAG service for support:", error);
      setSupportResponse("Could not connect to the Janani AI service.");
    } finally {
      setIsSupportLoading(false);
    }
  };

  return (
    <Motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      style={{
        position: 'fixed',
        top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0,0,0,0.6)',
        backdropFilter: 'blur(8px)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        zIndex: 2000,
        padding: '20px'
      }}
      onClick={onClose}
    >
      <Motion.div
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'white',
          padding: '2.5rem',
          borderRadius: '32px',
          maxWidth: '550px',
          width: '100%',
          boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <div style={{ width: '48px', height: '48px', background: 'var(--accent)', borderRadius: '12px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
              <Users color="var(--primary)" size={24} />
            </div>
            <h2 style={{ margin: 0, color: 'var(--text-dark)', fontSize: '1.5rem' }}>Janani Support</h2>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8' }}>
            <X size={24} />
          </button>
        </div>

        <div style={{ background: '#f8fafc', padding: '1.5rem', borderRadius: '20px', minHeight: '120px', maxHeight: '300px', overflowY: 'auto' }}>
          {supportResponse ? (
            <div style={{ display: 'flex', gap: '1rem' }}>
              <Bot color="var(--primary)" size={20} style={{ flexShrink: 0, marginTop: '4px' }} />
              <div>
                <p style={{ margin: 0, color: 'var(--text-dark)', lineHeight: '1.6', fontSize: '0.95rem' }}>{supportResponse}</p>
              </div>
            </div>
          ) : isSupportLoading ? (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
              <Loader size={24} className="animate-spin" color="var(--primary)" />
              <span style={{ marginLeft: '10px', color: 'var(--primary)', fontWeight: '500' }}>Thinking...</span>
            </div>
          ) : (
            <p style={{ color: '#64748b', textAlign: 'center', margin: '2rem 0' }}>
              How can we help you today? Type your query below and Janani AI will assist you using her medical knowledge.
            </p>
          )}
        </div>

        <div style={{ position: 'relative' }}>
          <input
            type="text"
            placeholder="Describe your concern..."
            value={supportInput}
            onChange={(e) => setSupportInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSupportSend()}
            disabled={isSupportLoading}
            style={{
              width: '100%',
              padding: '1rem 3.5rem 1rem 1.5rem',
              borderRadius: '50px',
              border: '2px solid var(--accent)',
              fontSize: '1rem',
              outline: 'none'
            }}
          />
          <button
            onClick={handleSupportSend}
            disabled={isSupportLoading || !supportInput.trim()}
            style={{
              position: 'absolute',
              right: '8px',
              top: '50%',
              transform: 'translateY(-50%)',
              background: 'var(--primary)',
              color: 'white',
              border: 'none',
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              opacity: (isSupportLoading || !supportInput.trim()) ? 0.5 : 1
            }}
          >
            <ArrowRight size={18} />
          </button>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem' }}>
          <button
            onClick={onProceed}
            disabled={loading}
            style={{
              flex: 1,
              padding: '0.8rem',
              borderRadius: '50px',
              border: '2px solid #ef4444',
              background: 'white',
              color: '#ef4444',
              fontWeight: '600',
              cursor: 'pointer',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '0.5rem',
              opacity: loading ? 0.7 : 1
            }}
          >
            <Phone size={18} />
            {loading ? 'Calling...' : 'Connect to Voice Support'}
          </button>
        </div>
      </Motion.div>
    </Motion.div>
  );
};

const Footer = ({ user }) => (
  <footer style={{ background: '#0f172a', color: 'white', padding: '4rem 0' }}>
    <div className="container">
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '3rem', marginBottom: '3rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <Heart color="white" size={24} fill="white" />
            <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>Janani</span>
          </div>
          <p style={{ opacity: 0.8, fontSize: '0.9rem' }}>
            Empowering Rural Women with RAG-Based Medical Intelligence
          </p>
        </div>
        <div>
          <h4 style={{ color: 'white', marginBottom: '1.5rem' }}>Quick Links</h4>
          <ul style={{ listStyle: 'none', opacity: 0.8 }}>
            <li style={{ marginBottom: '0.8rem', cursor: 'pointer' }} onClick={() => window.scrollToView('landing')}>About System</li>
            <li style={{ marginBottom: '0.8rem' }}>Research Base</li>
            {user && <li style={{ marginBottom: '0.8rem', cursor: 'pointer' }} onClick={() => window.scrollToView('dashboard')}>User Dashboard</li>}
            <li style={{ marginBottom: '0.8rem' }}>Terms of Care</li>
          </ul>
        </div>
        <div>
          <h4 style={{ color: 'white', marginBottom: '1.5rem' }}>Tech Stack</h4>
          <ul style={{ listStyle: 'none', opacity: 0.8 }}>
            <li style={{ marginBottom: '0.8rem' }}>Bhavini AI API</li>
            <li style={{ marginBottom: '0.8rem' }}>Google TTS</li>
            <li style={{ marginBottom: '0.8rem' }}>Hugging Face Models</li>
          </ul>
        </div>
      </div>
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '2rem', textAlign: 'center', opacity: 0.6, fontSize: '0.8rem' }}>
        © 2026 MatriCare AI. Hackathon Finalist Entry.
      </div>
    </div>
  </footer>
);


const ChatbotButton = ({ isVisible }) => {
  return (
    <AnimatePresence>
      {isVisible && (
        <ScrollLink
          to="voice-assistant"
          smooth={true}
          duration={800}
          offset={-70}
          style={{
            position: 'fixed',
            bottom: '2rem',
            right: '2rem',
            zIndex: 1000,
            cursor: 'pointer'
          }}
        >
          <Motion.div
            layoutId="shared-mic"
            initial={{ opacity: 0, scale: 0 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            whileHover={{ scale: 1.1, rotate: 10 }}
            whileTap={{ scale: 0.9 }}
            style={{
              width: '65px',
              height: '65px',
              borderRadius: '50%',
              background: 'var(--primary)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              boxShadow: '0 10px 30px rgba(176, 24, 84, 0.4)',
              color: 'white'
            }}
          >
            <Bot size={35} color="white" />
          </Motion.div>
        </ScrollLink>
      )}
    </AnimatePresence>
  );
};


function App() {
  const [view, setView] = useState('landing');
  const [showAuth, setShowAuth] = useState(false);
  const [showContact, setShowContact] = useState(false);
  const [contactLoading, setContactLoading] = useState(false);
  const assistantRef = React.useRef(null);
  const [assistantVisible, setAssistantVisible] = useState(false);
  const [toast, setToast] = useState(null);

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 4000);
  };

  // Extend window object to allow navigation from components
  useEffect(() => {
    window.scrollToView = (v) => {
      setView(v);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };
  }, []);

  const [user, setUser] = useState(() => {
    const savedUser = localStorage.getItem('user');
    return savedUser ? JSON.parse(savedUser) : null;
  });

  useEffect(() => {
    const handleScroll = () => {
      if (assistantRef.current) {
        const rect = assistantRef.current.getBoundingClientRect();
        // Hide chatbot when Assistant is entering the middle of the screen
        setAssistantVisible(rect.top < window.innerHeight * 0.6);
      }
    };
    handleScroll();
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleTriggerCall = async () => {
    setContactLoading(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/api/voice/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.message || 'Failed to trigger call');
      setShowContact(false); // Close modal on success
      showToast('Call initiated! You will receive a call shortly.', 'success');
    } catch (err) {
      console.error('Error triggering call:', err);
      showToast('Could not initiate call. Please check if the backend is running.', 'error');
    } finally {
      setContactLoading(false);
    }
  };

  useEffect(() => {
    // Initial user state is now handled by useState initializer
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
  };

  return (
    <div className="App">
      <ChatbotButton isVisible={!assistantVisible} />
      <Navbar
        onAuthClick={() => setShowAuth(true)}
        user={user}
        onLogout={handleLogout}
        onContactClick={() => setShowContact(true)}
        contactLoading={contactLoading}
        setView={setView}
        currentView={view}
      />

      <AnimatePresence mode="wait">
        {view === 'landing' ? (
          <Motion.div key="landing">
            <Hero onAuthClick={() => setShowAuth(true)} user={user} setView={setView} />
            <DemoSection />
            <MainContent assistantRef={assistantRef} user={user} />
          </Motion.div>
        ) : (
          <Dashboard
            key="dashboard"
            user={user}
            onBack={() => setView('landing')}
            onEmergencyCall={handleTriggerCall}
          />
        )}
      </AnimatePresence>

      <Footer user={user} />

      <AnimatePresence>
        {showAuth && (
          <Auth
            onClose={() => setShowAuth(false)}
            onAuthSuccess={(userData) => setUser(userData)}
          />
        )}
        {showContact && (
          <ContactModal
            onClose={() => setShowContact(false)}
            onProceed={handleTriggerCall}
            loading={contactLoading}
            user={user}
            selectedLanguage={view === 'landing' ? 'hi-IN' : 'en-IN'} // Default to Hindi for landing, English for dashboard
          />
        )}
      </AnimatePresence>

      {/* Toast Notification */}
      <AnimatePresence>
        {toast && (
          <Motion.div
            initial={{ opacity: 0, y: -60 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -60 }}
            transition={{ type: 'spring', damping: 20, stiffness: 300 }}
            style={{
              position: 'fixed',
              top: '80px',
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 9999,
              background: toast.type === 'error' ? '#fef2f2' : '#f0fdf4',
              color: toast.type === 'error' ? '#dc2626' : '#16a34a',
              border: `1px solid ${toast.type === 'error' ? '#fecaca' : '#bbf7d0'}`,
              padding: '0.8rem 1.5rem',
              borderRadius: '12px',
              fontWeight: '600',
              fontSize: '0.95rem',
              boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              maxWidth: '90vw'
            }}
          >
            <span>{toast.type === 'error' ? '⚠' : '✓'}</span>
            {toast.message}
            <button
              onClick={() => setToast(null)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', marginLeft: '0.5rem', fontSize: '1.1rem', color: 'inherit', opacity: 0.6 }}
            >
              ×
            </button>
          </Motion.div>
        )}
      </AnimatePresence>

      {/* Embedded Animations Styles */}
      <style>{`
        .hover-text-primary:hover {
          color: var(--primary) !important;
        }
      `}</style>
    </div>
  );
}


export default App;
