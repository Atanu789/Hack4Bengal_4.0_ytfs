import { useState, useEffect, useRef } from 'react';
import { Clock,Search, Brain, FileText, Settings, Zap, RefreshCw, Play, Sparkles, Eye, Layers, ChevronRight, Activity, Cpu, Globe, Star, Volume2, Mic, Target, Lightbulb, Network, Shield, Wand2, MicOff, Maximize2, Filter, X } from 'lucide-react';

// Chrome extension type declarations
declare global {
  interface Window {
  
    SpeechRecognition: typeof SpeechRecognition;
    webkitSpeechRecognition: typeof SpeechRecognition;
    webkitAudioContext: typeof AudioContext;
  }
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onstart: ((this: SpeechRecognition, ev: Event) => any) | null;
  onresult: ((this: SpeechRecognition, ev: SpeechRecognitionEvent) => any) | null;
  onerror: ((this: SpeechRecognition, ev: SpeechRecognitionErrorEvent) => any) | null;
  onend: ((this: SpeechRecognition, ev: Event) => any) | null;
  start(): void;
  stop(): void;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  readonly length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  readonly length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: 'no-speech' | 'aborted' | 'audio-capture' | 'network' | 'not-allowed' | 'service-not-allowed' | 'bad-grammar' | 'language-not-supported';
  message?: string;
}

declare var SpeechRecognition: {
  prototype: SpeechRecognition;
  new (): SpeechRecognition;
};

function App() {
  const [videoLink, setVideoLink] = useState('');
  const [searchKeyword, setSearchKeyword] = useState('');
  const [semanticSearch, setSemanticSearch] = useState(false);
  const [suggestion, setSuggestion] = useState(false);
  const [advancedSearch, setAdvancedSearch] = useState(false);
  const [isVideoDetected, setIsVideoDetected] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [searchProgress, setSearchProgress] = useState(0);
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isSupported, setIsSupported] = useState<boolean>(false);
  const [transcript, setTranscript] = useState<string>('');
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const extensionRef = useRef<HTMLDivElement>(null);
const [searchHistory, setSearchHistory] = useState<Array<{
  videoLink: string;
  keyword: string;
  timestamp: number;
  results?: any[];
}>>([]);
  // Prevent extension from closing when clicking inside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (extensionRef.current && !extensionRef.current.contains(event.target as Node)) {
        event.stopPropagation();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Speech recognition setup
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      setIsSupported(true);
      
      const SpeechRecognitionConstructor = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognitionConstructor();
      
      if (recognitionRef.current) {
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = true;
        recognitionRef.current.lang = 'en-US';
        recognitionRef.current.maxAlternatives = 1;

        recognitionRef.current.onstart = () => {
          setIsListening(true);
          setTranscript('');
        };

        recognitionRef.current.onresult = (event: SpeechRecognitionEvent) => {
          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
              finalTranscript += transcript;
            } else {
              interimTranscript += transcript;
            }
          }

          const currentTranscript = finalTranscript || interimTranscript;
          setTranscript(currentTranscript);
          
          if (finalTranscript) {
            setSearchKeyword(finalTranscript.trim());
            setIsListening(false);
          }
        };

        recognitionRef.current.onerror = (event: SpeechRecognitionErrorEvent) => {
          setIsListening(false);
          setTranscript('');
          
          switch(event.error) {
            case 'not-allowed':
              alert('Microphone access denied. Please allow microphone access and try again.');
              break;
            case 'no-speech':
              console.log('No speech detected');
              break;
            case 'network':
              alert('Network error occurred. Please check your connection.');
              break;
            default:
              console.log('Speech recognition error:', event.error);
          }
        };

        recognitionRef.current.onend = () => {
          setIsListening(false);
        };
      }
    } else {
      setIsSupported(false);
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  // Video detection
  useEffect(() => {
    const detectVideoLink = () => {
      try {
        if (typeof window !== 'undefined' && window.chrome && window.chrome.tabs) {
          window.chrome.tabs.query({ active: true, currentWindow: true }, (tabs: any[]) => {
            const currentUrl = tabs[0]?.url || '';
            
            if (currentUrl.includes('youtube.com/watch') || 
                currentUrl.includes('youtu.be/') ||
                currentUrl.includes('vimeo.com/') ||
                currentUrl.includes('dailymotion.com/') ||
                currentUrl.includes('twitch.tv/')) {
              setVideoLink(currentUrl);
              setIsVideoDetected(true);
            } else {
              setVideoLink('');
              setIsVideoDetected(false);
            }
          });
        } else {
          const currentUrl = window.location.href;
          
          if (currentUrl.includes('youtube.com/watch') || 
              currentUrl.includes('youtu.be/') ||
              currentUrl.includes('vimeo.com/') ||
              currentUrl.includes('dailymotion.com/') ||
              currentUrl.includes('twitch.tv/')) {
            setVideoLink(currentUrl);
            setIsVideoDetected(true);
          } else {
            const videoElements = document.querySelectorAll('video, iframe[src*="youtube"], iframe[src*="vimeo"]');
            if (videoElements.length > 0) {
              setVideoLink(currentUrl);
              setIsVideoDetected(true);
            } else {
              setVideoLink('');
              setIsVideoDetected(false);
            }
          }
        }
      } catch (error) {
        console.error('Error detecting video:', error);
        setVideoLink('');
        setIsVideoDetected(false);
      }
    };

    detectVideoLink();
  }, []);

  // Search progress animation
  useEffect(() => {
    if (isSearching) {
      const stages = [
        { progress: 15, duration: 300 },
        { progress: 35, duration: 500 },
        { progress: 60, duration: 800 },
        { progress: 85, duration: 600 },
        { progress: 100, duration: 400 }
      ];

      let currentStage = 0;
      const processStage = () => {
        if (currentStage < stages.length) {
          const stage = stages[currentStage];
          setTimeout(() => {
            setSearchProgress(stage.progress);
            currentStage++;
            processStage();
          }, stage.duration);
        }
      };
      processStage();
    } else {
      setSearchProgress(0);
    }
  }, [isSearching]);
useEffect(() => {
  // Load search history from local storage
  const storedHistory = localStorage.getItem('smartSeekSearchHistory');
  if (storedHistory) {
    try {
      setSearchHistory(JSON.parse(storedHistory));
    } catch (e) {
      console.error('Failed to parse search history', e);
    }
  }
}, []);
 const searchAutoVideos = async (videoUrl: string, searchQuery: string,limit:number=5) => {
    try {
      const response = await fetch('http://localhost:8000/youtube/video/suggestion', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: videoUrl,
          query: searchQuery,
          limit
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error searching video:', error);
      throw error;
    }
  };
  const searchVideo = async (videoUrl: string, searchQuery: string) => {
    try {
      const response = await fetch('http://localhost:8000/youtube/video/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: videoUrl,
          query: searchQuery,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error searching video:', error);
      throw error;
    }
  };

  const handleSearchKeywordChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchKeyword(e.target.value);
  };

  const handleSearch = async () => {
  if (!videoLink.trim()) {
    alert('No video detected on this page!');
    return;
  }
  if (!searchKeyword.trim()) {
    alert('Please enter a search keyword!');
    return;
  }

  const activeSearchTypes: string[] = [];
  if (semanticSearch) activeSearchTypes.push('Semantic Search');
  if (suggestion) activeSearchTypes.push('Suggestion');
  if (advancedSearch) activeSearchTypes.push('Advanced Search');

  if (activeSearchTypes.length === 0) {
    alert('Please select at least one search method!');
    return;
  }
  if(suggestion && semanticSearch){
    try {
    const results = await searchAutoVideos(videoLink, searchKeyword, 3);
    console.log('Auto-suggest results:', results);
    // Handle both current video results and similar videos
    setSearchResults(results.current_video_results || []);
    // You might want to do something with results.similar_videos too
  } catch (error) {
    console.error('Auto-suggest error:', error);
  }

  }

  setIsSearching(true);
  setApiError(null);
  setSearchResults([]);
  
  try {
    const results = await searchVideo(videoLink, searchKeyword);
    
    // Create new search entry
    const newSearchEntry = {
      videoLink,
      keyword: searchKeyword,
      timestamp: Date.now(),
      results
    };

    // Update state and local storage
    setSearchHistory(prev => {
      const updatedHistory = [newSearchEntry, ...prev].slice(0, 10); // Keep last 10 searches
      localStorage.setItem('smartSeekSearchHistory', JSON.stringify(updatedHistory));
      return updatedHistory;
    });

    setSearchResults(results);
    setShowResults(true);
  } catch (error) {
    console.error('Search error:', error);
    setApiError(`An error occurred while searching. Please try again. ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    setIsSearching(false);
  }
};

  const refreshVideoDetection = () => {
    try {
      if (typeof window !== 'undefined' && window.chrome && window.chrome.tabs) {
        window.chrome.tabs.query({ active: true, currentWindow: true }, (tabs: any[]) => {
          const currentUrl = tabs[0]?.url || '';
          
          if (currentUrl.includes('youtube.com/watch') || 
              currentUrl.includes('youtu.be/') ||
              currentUrl.includes('vimeo.com/') ||
              currentUrl.includes('dailymotion.com/') ||
              currentUrl.includes('twitch.tv/')) {
            setVideoLink(currentUrl);
            setIsVideoDetected(true);
          } else {
            setVideoLink('');
            setIsVideoDetected(false);
          }
        });
      } else {
        const currentUrl = window.location.href;
        if (currentUrl.includes('youtube.com/watch') || 
            currentUrl.includes('youtu.be/') ||
            currentUrl.includes('vimeo.com/') ||
            currentUrl.includes('dailymotion.com/') ||
            currentUrl.includes('twitch.tv/')) {
          setVideoLink(currentUrl);
          setIsVideoDetected(true);
        } else {
          const videoElements = document.querySelectorAll('video, iframe[src*="youtube"], iframe[src*="vimeo"]');
          if (videoElements.length > 0) {
            setVideoLink(currentUrl);
            setIsVideoDetected(true);
          } else {
            setVideoLink('');
            setIsVideoDetected(false);
          }
        }
      }
    } catch (error) {
      console.error('Error refreshing video detection:', error);
      setVideoLink('');
      setIsVideoDetected(false);
    }
  };

  const toggleVoiceSearch = () => {
    if (!isSupported) return;
    
    if (isListening) {
      recognitionRef.current?.stop();
    } else {
      recognitionRef.current?.start();
    }
  };

  const SearchMethodCard = ({ icon: Icon, title, description, active, onClick, gradient, features }: any) => (
    <div 
      className={`group relative cursor-pointer transition-all duration-500 transform hover:scale-[1.01] ${
        active ? 'scale-[1.01]' : ''
      }`}
      onClick={onClick}
    >
      {/* Search History Section */}
<div className="space-y-3">
  <h2 className="text-sm font-bold text-white flex items-center space-x-2">
    <Clock className="w-4 h-4 text-purple-200" />
    <span>Recent Searches</span>
    <div className="flex-1 h-px bg-gradient-to-r from-purple-400/60 via-blue-400/40 to-transparent"></div>
  </h2>

  {searchHistory.length > 0 ? (
    <div className="space-y-2 max-h-40 overflow-y-auto custom-scrollbar pr-2">
      {searchHistory.map((search, index) => (
        <div 
          key={index}
          className="group relative p-2 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-xl 
            border border-white/20 rounded-lg hover:border-purple-400/60 transition-all duration-300"
          onClick={() => {
            setVideoLink(search.videoLink);
            setSearchKeyword(search.keyword);
            if (search.results) {
              setSearchResults(search.results);
              setShowResults(true);
            }
          }}
        >
          <div className="flex items-start space-x-2">
            <div className="w-2 h-2 mt-1.5 bg-purple-400 rounded-full flex-shrink-0"></div>
            <div className="flex-1 overflow-hidden">
              <p className="text-xs text-white/90 truncate">{search.keyword}</p>
              <div className="flex items-center justify-between mt-1">
                <span className="text-[10px] text-purple-300 font-medium">
                  {new Date(search.timestamp).toLocaleString()}
                </span>
                <span className="text-[10px] text-gray-400 truncate max-w-[100px]">
                  {search.videoLink.replace(/^https?:\/\/(www\.)?/, '')}
                </span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  ) : (
    <div className="text-center p-2 bg-white/5 rounded-lg border border-white/10 text-gray-400 text-xs">
      No search history yet
    </div>
  )}
</div>
      <div className={`relative backdrop-blur-2xl bg-gradient-to-br from-white/15 to-white/5 
        rounded-xl border border-white/30 p-3 overflow-hidden transition-all duration-500
        ${active ? 'ring-1 ring-white/40 shadow-lg shadow-purple-500/30 border-white/50' : 'hover:border-white/40 hover:shadow-md hover:shadow-purple-500/20'}
        ${active ? 'bg-gradient-to-br from-white/25 to-white/10' : ''}
      `}>
        <div className={`absolute inset-0 opacity-0 transition-all duration-500 ${
          active ? 'opacity-50' : 'group-hover:opacity-60'
        } bg-gradient-to-br ${gradient} blur-xl`}></div>
        
        <div className={`absolute inset-0 opacity-0 transition-all duration-700 ${
          active ? 'opacity-30' : 'group-hover:opacity-20'
        }`}>
          <div className={`absolute top-1 right-1 w-12 h-12 bg-gradient-to-br ${gradient} rounded-full blur-xl animate-pulse`}></div>
        </div>
        
        <div className="relative z-10 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`relative w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-500 overflow-hidden
                ${active ? `bg-gradient-to-br ${gradient} shadow-lg shadow-purple-500/40` : 'bg-white/15 group-hover:bg-white/25 backdrop-blur-xl'}
              `}>
                <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 transition-opacity duration-200 ${
                  active ? 'opacity-20' : 'group-hover:opacity-50'
                } blur-md`}></div>
                <Icon className={`relative w-5 h-5 transition-all duration-500 ${
                  active ? 'text-white drop-shadow-md' : 'text-gray-300 group-hover:text-white'
                }`} />
                {active && (
                  <div className="absolute inset-0 border border-white/30 rounded-xl animate-spin" style={{animationDuration: '3s'}}></div>
                )}
              </div>
              <div className="space-y-0.5">
                <h3 className={`font-bold text-sm transition-all duration-500 ${
                  active ? 'text-white drop-shadow-md' : 'text-gray-200 group-hover:text-white'
                }`}>{title}</h3>
                <p className={`text-xs transition-all duration-500 ${
                  active ? 'text-white/90' : 'text-gray-400 group-hover:text-gray-300'
                }`}>{description}</p>
              </div>
            </div>
            
            <div className={`relative w-5 h-5 rounded-full border-2 transition-all duration-500 flex items-center justify-center overflow-hidden ${
              active 
                ? 'border-white bg-white shadow-lg shadow-white/50' 
                : 'border-gray-400 group-hover:border-white/70 backdrop-blur-xl'
            }`}>
              {active && (
                <>
                  <div className={`w-3 h-3 rounded-full bg-gradient-to-br ${gradient} animate-pulse`}></div>
                  <div className="absolute inset-0 bg-white/20 rounded-full animate-ping"></div>
                </>
              )}
            </div>
          </div>
          
          {active && features && (
            <div className="flex flex-wrap gap-1 pt-1 border-t border-white/20">
              {features.map((feature: string, index: number) => (
                <div 
                  key={index} 
                  className="px-1.5 py-0.5 bg-white/20 backdrop-blur-xl rounded-md text-[10px] text-white/90 border border-white/30 shadow-md"
                >
                  {feature}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  const hasActiveSearchMethods = semanticSearch || suggestion || advancedSearch;

  return (
    <div 
      ref={extensionRef}
      className="w-[380px] h-[830px] flex flex-col bg-gradient-to-br from-slate-900 via-indigo-950 to-purple-950 relative overflow-hidden rounded-xl"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,_var(--tw-gradient-stops))] from-purple-900/30 via-slate-900/60 to-black/80"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-blue-900/40 via-indigo-900/30 to-transparent"></div>
      <div className="absolute inset-0 bg-[conic-gradient(from_0deg_at_50%_50%,_transparent_0deg,_purple-500/10_60deg,_transparent_120deg,_blue-500/10_180deg,_transparent_240deg,_pink-500/10_300deg,_transparent_360deg)] animate-spin" style={{animationDuration: '20s'}}></div>
      
      <div className="relative flex flex-col h-full backdrop-blur-2xl bg-gradient-to-b from-black/40 via-black/30 to-black/40 border border-white/20 shadow-2xl rounded-xl m-1 overflow-hidden">
        {/* Header */}
        <div className="relative bg-gradient-to-r from-purple-600/25 via-blue-600/25 to-pink-600/25 backdrop-blur-3xl border-b border-white/20 px-4 py-3 flex-shrink-0">
          <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
              <div className="relative group">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-500 via-blue-500 to-pink-500 rounded-xl flex items-center justify-center shadow-lg shadow-purple-500/40 transition-all duration-500 group-hover:scale-105">
                  <Sparkles className="w-5 h-5 text-white animate-pulse drop-shadow-md" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full shadow-md overflow-hidden">
                  <div className="absolute inset-0 bg-green-400 rounded-full animate-ping opacity-60"></div>
                  <div className="absolute inset-0.5 bg-white rounded-full animate-pulse"></div>
                </div>
                <div className="absolute inset-0 border border-white/20 rounded-xl animate-spin opacity-0 group-hover:opacity-100 transition-opacity duration-500" style={{animationDuration: '4s'}}></div>
              </div>
              <div className="space-y-0.5">
                <h1 className="text-lg font-black bg-gradient-to-r from-white via-purple-200 to-pink-200 bg-clip-text text-transparent drop-shadow-md">
                  Smart Seek
                </h1>
                <p className="text-xs text-gray-300/90 flex items-center space-x-1">
                  <span>Find What Matters, Instantly!</span>
                  <Network className="w-3 h-3 text-blue-400" />
                </p>
              </div>
            </div>
            <div className="flex flex-col items-end space-y-1">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-red-400 rounded-full animate-pulse shadow-md shadow-red-400/50"></div>
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse shadow-md shadow-green-400/50" style={{animationDelay: '0.5s'}}></div>
                <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse shadow-md shadow-blue-400/50" style={{animationDelay: '1s'}}></div>
              </div>
              <div className="flex items-center space-x-1 text-[12px] text-gray-400">
                <Shield className="text-yellow-200 w-2.5 h-2.5" />
                <span>Secure</span>
              </div>
            </div>
          </div>
        </div>
 
        <div className="flex-1 overflow-y-auto px-4 py-2 space-y-3.5 custom-scrollbar">
          {/* Video Detector Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white flex items-center space-x-2">
                <Eye className="w-4 h-4 text-purple-200" />
                <span>Video Detector</span>
                <div className="flex-1 h-px bg-gradient-to-r from-purple-400/60 via-blue-400/40 to-transparent"></div>
              </h2>
              <button
                onClick={refreshVideoDetection}
                className="group flex items-center space-x-1 px-2 py-1 rounded-lg bg-white/10 hover:bg-white/20 
                  transition-all duration-500 border border-white/20 hover:border-white/40 hover:scale-105 backdrop-blur-xl
                  shadow-md hover:shadow-lg hover:shadow-blue-500/20 text-xs"
              >
                <RefreshCw className="w-3.5 h-3.5 text-blue-400 group-hover:rotate-180 transition-transform duration-700" />
                <span className="text-gray-300 group-hover:text-white font-medium">Refresh</span>
              </button>
            </div>
            
            <div className={`relative backdrop-blur-2xl rounded-xl border overflow-hidden transition-all duration-500 group ${
              isVideoDetected 
                ? 'bg-gradient-to-br from-emerald-500/25 to-green-500/20 border-emerald-400/40 shadow-lg shadow-emerald-500/30' 
                : 'bg-gradient-to-br from-red-500/25 to-pink-500/20 border-red-400/40 shadow-lg shadow-red-500/30'
            }`}>
              <div className="absolute top-0 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-white/30 to-transparent"></div>
              
              <div className={`absolute inset-0 opacity-10 ${
                isVideoDetected ? 'bg-green-400' : 'bg-red-400'
              }`} style={{
                backgroundImage: 'radial-gradient(circle at 25% 25%, white 2px, transparent 2px)',
                backgroundSize: '20px 20px',
                animation: 'movePattern 20s linear infinite'
              }}></div>
              
              <div className="relative p-3 space-y-2">
                <div className="flex items-center space-x-3">
                  <div className="relative">
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center shadow-lg transition-all duration-500 ${
                      isVideoDetected ? 'bg-emerald-500' : 'bg-red-500'
                    }`}>
                      {isVideoDetected ? (
                        <Play className="w-3 h-3 text-white drop-shadow-md" />
                      ) : (
                        <Globe className="w-3 h-3 text-white drop-shadow-md" />
                      )}
                    </div>
                    <div className={`absolute inset-0 rounded-full animate-ping opacity-30 ${
                      isVideoDetected ? 'bg-emerald-400' : 'bg-red-400'
                    }`}></div>
                    <div className={`absolute -inset-0.5 rounded-full animate-ping opacity-20 ${
                      isVideoDetected ? 'bg-emerald-400' : 'bg-red-400'
                    }`} style={{animationDelay: '0.5s'}}></div>
                  </div>
                  <div className="flex-1 space-y-0.5">
                    <h3 className={`font-bold text-sm ${
                      isVideoDetected ? 'text-emerald-100' : 'text-red-100'
                    }`}>
                      {isVideoDetected ? 'Video Detected' : 'No Video'}
                    </h3>
                    <p className={`text-xs ${
                      isVideoDetected ? 'text-emerald-200/90' : 'text-red-200/90'
                    }`}>
                      {isVideoDetected ? 'Ready for analysis' : 'Navigate to video'}
                    </p>
                  </div>
                  {isVideoDetected && (
                    <div className="flex flex-col items-center space-y-0.5">
                      <div className="flex items-center space-x-0.5">
                        <Star className="w-3 h-3 text-yellow-400 animate-pulse" />
                      </div>
                      <span className="text-[12px] text-gray-300 font-medium">Premium</span>
                    </div>
                  )}
                </div>
                
                {isVideoDetected && (
                  <div className="relative p-2 rounded-lg bg-black/40 border border-white/20 backdrop-blur-xl overflow-hidden">
                    <div className="flex items-center space-x-2">
                      <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
                      <p className="text-xs text-gray-300 font-mono leading-relaxed break-all">
                        {videoLink.length > 40 ? `${videoLink.substring(0, 40)}...` : videoLink}
                      </p>
                    </div>
                    <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-green-400 to-transparent animate-pulse"></div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Search Query Section */}
          <div className="space-y-3">
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <Search className="w-4 h-4 text-blue-200" />
              <span>Search Query</span>
              <div className="flex-1 h-px bg-gradient-to-r from-blue-400/60 via-purple-400/40 to-transparent"></div>
              <Wand2 className="w-3 h-3 text-purple-400" />
            </h2>
            
            <div className="relative group">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/30 via-purple-500/30 to-pink-500/30 rounded-xl blur opacity-0 group-hover:opacity-100 transition-all duration-500"></div>
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg opacity-0 group-focus-within:opacity-100 transition-opacity duration-500"></div>
              
              <input
                type="text"
                value={searchKeyword}
                onChange={handleSearchKeywordChange}
                placeholder={isListening ? "Listening..." : "What to search in video..."}
                className="relative w-full px-3 py-2 pl-10 pr-20 bg-gradient-to-r from-white/15 to-white/10 backdrop-blur-2xl 
                  border border-white/30 rounded-xl focus:outline-none focus:ring-1 focus:ring-blue-500/60 
                  focus:border-blue-400/60 placeholder-gray-400 text-white transition-all duration-500 
                  hover:bg-white/20 hover:border-white/40 text-sm font-medium shadow-lg hover:shadow-blue-500/20
                  focus:shadow-lg focus:shadow-purple-500/30"
              />
              
              <div className="absolute left-3 top-1/2 transform -translate-y-1/2">
                <Search className="w-4 h-4 text-gray-400 group-hover:text-blue-400 group-focus-within:text-purple-400 transition-colors duration-500" />
              </div>
              
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 flex items-center space-x-2">
                <button
                  onClick={toggleVoiceSearch}
                  disabled={!isSupported}
                  className={`p-1 rounded-l transition-all duration-300 ${
                    isListening
                      ? 'bg-red-500/20 text-red-400 animate-pulse shadow-lg shadow-red-500/30'
                      : isSupported
                      ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30 hover:scale-110 shadow-lg hover:shadow-green-500/30'
                      : 'bg-gray-500/20 text-gray-500 cursor-not-allowed'
                  }`}
                  title={isListening ? "Stop listening" : "Start voice search"}
                >
                  {isListening ? (
                    <div className="relative">
                      <MicOff className="w-4 h-4" />
                      <div className="absolute -inset-1 border-2 border-red-400 rounded-full animate-ping"></div>
                    </div>
                  ) : (
                    <Mic className="w-4 h-4" />
                  )}
                </button>
                
                <div className="w-px h-4 bg-gradient-to-t from-purple-400/50 to-transparent"></div>
                
                <div className="flex items-center space-x-1">
                  <Cpu className="w-4 h-4 text-purple-400 animate-pulse" />
                  <div className="w-0.5 h-4 bg-gradient-to-t from-purple-400 to-transparent rounded-full animate-pulse"></div>
                </div>
              </div>
            </div>
            
            {isListening && (
              <div className="flex items-center justify-center space-x-2 text-sm text-green-400">
                <Volume2 className="w-4 h-4 animate-pulse" />
                <span>Listening... {transcript && `"${transcript}"`}</span>
                <div className="flex space-x-0.5">
                  <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                  <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                </div>
              </div>
            )}
            
            {!isSupported && (
              <div className="text-xs text-amber-400 text-center bg-amber-500/10 border border-amber-500/20 rounded-lg p-2">
                Voice search requires Chrome, Edge, or Safari browser
              </div>
            )}
            
            {isSupported && (
              <div className="text-xs text-blue-300/70 text-center">
                Allow microphone access to use voice search...
              </div>
            )}
          </div>

          {/* Search Methods Section */}
          <div className="space-y-3">
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <Brain className="w-4 h-4 text-purple-200" />
              <span>Search Methods</span>
              <div className="flex-1 h-px bg-gradient-to-r from-purple-400/60 via-pink-400/40 to-transparent"></div>
              <Target className="w-3 h-3 text-pink-400" />
            </h2>
            
            <div className="space-y-3">
              <SearchMethodCard
                icon={Brain}
                title="Semantic Search"
                description="Understands meaning beyond keywords"
                active={semanticSearch}
                onClick={() => setSemanticSearch(!semanticSearch)}
                gradient="from-purple-500 via-violet-500 to-pink-500"
                features={["Context-aware", "AI-powered", "Natural language"]}
              />
              
              <SearchMethodCard
                icon={FileText}
                title="Suggestion"
                description="Get related content suggestions"
                active={suggestion}
                onClick={() => setSuggestion(!suggestion)}
                gradient="from-emerald-500 via-green-500 to-teal-500"
                features={["Related topics", "Content ideas", "Expand search"]}
              />
              
              <SearchMethodCard
                icon={Settings}
                title="Advanced Search"
                description="Precise control over search parameters"
                active={advancedSearch}
                onClick={() => setAdvancedSearch(!advancedSearch)}
                gradient="from-orange-500 via-red-500 to-pink-500"
                features={["Fine-tuned", "Filters", "Custom parameters"]}
              />
            </div>
          </div>

          {hasActiveSearchMethods && (
            <div className="relative backdrop-blur-2xl bg-gradient-to-br from-blue-500/25 via-purple-500/20 to-pink-500/25 
              border border-blue-400/40 rounded-xl p-3 shadow-lg shadow-blue-500/30 transition-all duration-500 overflow-hidden">
              
              <div className="absolute inset-0 opacity-5 bg-gradient-to-r from-blue-400 to-purple-400" style={{
                backgroundImage: 'radial-gradient(circle at 20% 50%, white 2px, transparent 2px), radial-gradient(circle at 80% 50%, white 2px, transparent 2px)',
                backgroundSize: '25px 25px',
                animation: 'movePattern 15s linear infinite'
              }}></div>
              
              <div className="relative space-y-2">
                <div className="flex items-center space-x-2">
                  <div className="relative">
                    <div className="w-2.5 h-2.5 bg-blue-400 rounded-full animate-pulse shadow-md shadow-blue-400/50"></div>
                    <div className="absolute inset-0 bg-blue-400 rounded-full animate-ping opacity-30"></div>
                  </div>
                  <h3 className="font-bold text-sm text-blue-100">AI Active</h3>
                  <div className="flex-1 h-px bg-gradient-to-r from-blue-400/60 via-purple-400/40 to-transparent"></div>
                  <Layers className="w-4 h-4 text-blue-400 animate-pulse" />
                  <Activity className="w-4 h-4 text-green-400 animate-bounce" />
                </div>
                
                <div className="grid grid-cols-2 gap-2">
                  {semanticSearch && (
                    <div className="group relative px-2 py-1.5 bg-gradient-to-r from-purple-500/30 via-violet-500/25 to-pink-500/30 
                      text-purple-100 text-xs rounded-lg border border-purple-400/40 
                      flex items-center space-x-1 shadow-md backdrop-blur-xl hover:scale-105 transition-all duration-500">
                      <div className="absolute inset-0 bg-gradient-to-r from-purple-400/20 to-pink-400/20 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      <div className="relative w-1.5 h-1.5 bg-purple-400 rounded-full animate-pulse shadow-md shadow-purple-400/50"></div>
                      <span className="relative font-semibold">Semantic</span>
                      <Brain className="relative w-3 h-3 ml-auto" />
                    </div>
                  )}
                  {suggestion && (
                    <div className="group relative px-2 py-1.5 bg-gradient-to-r from-emerald-500/30 via-green-500/25 to-teal-500/30 
                      text-emerald-100 text-xs rounded-lg border border-emerald-400/40 
                      flex items-center space-x-1 shadow-md backdrop-blur-xl hover:scale-105 transition-all duration-500">
                      <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/20 to-teal-400/20 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      <div className="relative w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse shadow-md shadow-emerald-400/50"></div>
                      <span className="relative font-semibold">Suggestion</span>
                      <FileText className="relative w-3 h-3 ml-auto" />
                    </div>
                  )}
                  {advancedSearch && (
                    <div className="group relative px-2 py-1.5 bg-gradient-to-r from-orange-500/30 via-red-500/25 to-pink-500/30 
                      text-orange-100 text-xs rounded-lg border border-orange-400/40 
                      flex items-center space-x-1 shadow-md backdrop-blur-xl hover:scale-105 transition-all duration-500 col-span-2">
                      <div className="absolute inset-0 bg-gradient-to-r from-orange-400/20 to-red-400/20 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      <div className="relative w-1.5 h-1.5 bg-orange-400 rounded-full animate-pulse shadow-md shadow-orange-400/50"></div>
                      <span className="relative font-semibold">Advanced AI</span>
                      <Settings className="relative w-3 h-3 ml-auto" />
                    </div>
                  )}
                </div>
                
                <div className="flex items-center justify-between pt-1 border-t border-white/20 text-xs">
                  <div className="flex items-center space-x-1 text-gray-300">
                    <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
                    <span>Optimal</span>
                  </div>
                  <div className="flex items-center space-x-1 text-gray-300">
                    <span>98.7%</span>
                    <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* AI Insights Section */}
          <div className="space-y-3">
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <Lightbulb className="w-4 h-4 text-yellow-200" />
              <span>AI Insights</span>
              <div className="flex-1 h-px bg-gradient-to-r from-yellow-400/60 via-orange-400/40 to-transparent"></div>
            </h2>
            
            <div className="grid grid-cols-2 gap-3">
              <div className="relative group p-2 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-2xl 
                border border-white/20 rounded-lg hover:border-white/40 transition-all duration-500 overflow-hidden">
                
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                
                <div className="relative space-y-1">
                  <div className="flex items-center space-x-1">
                    <Volume2 className="w-3 h-3 text-blue-400" />
                    <span className="text-xs font-semibold text-white">Audio</span>
                  </div>
                  <div className="flex items-center space-x-1 mt-1">
                    <div className="flex-1 h-0.5 bg-white/20 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-blue-400 to-purple-400 rounded-full animate-pulse" style={{width: '75%'}}></div>
                    </div>
                    <span className="text-[10px] text-blue-400 font-medium">75%</span>
                  </div>
                </div>
              </div>
              
              <div className="relative group p-2 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-2xl 
                border border-white/20 rounded-lg hover:border-white/40 transition-all duration-500 overflow-hidden">
                
                <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-emerald-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                
                <div className="relative space-y-1">
                  <div className="flex items-center space-x-1">
                    <Eye className="w-3 h-3 text-green-400" />
                    <span className="text-xs font-semibold text-white">Visual</span>
                  </div>
                  <div className="flex items-center space-x-1 mt-1">
                    <div className="flex-1 h-0.5 bg-white/20 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-green-400 to-emerald-400 rounded-full animate-pulse" style={{width: '92%'}}></div>
                    </div>
                    <span className="text-[10px] text-green-400 font-medium">92%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions Section */}
          <div className="space-y-3">
            <h2 className="text-sm font-bold text-white flex items-center space-x-2">
              <Filter className="w-4 h-4 text-cyan-200" />
              <span>Quick Actions</span>
              <div className="flex-1 h-px bg-gradient-to-r from-cyan-400/60 via-blue-400/40 to-transparent"></div>
            </h2>
            
            <div className="grid grid-cols-2 gap-2">
              <button className="group relative p-2 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-2xl 
                border border-white/20 rounded-lg hover:border-white/40 transition-all duration-500 
                hover:scale-105 hover:shadow-md hover:shadow-cyan-500/20 overflow-hidden text-xs">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="relative flex items-center space-x-1">
                  <Mic className="w-3 h-3 text-cyan-400 group-hover:animate-pulse" />
                  <span className="font-medium text-white">Suggestion</span>
                </div>
              </button>
              
              <button className="group relative p-2 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-2xl 
                border border-white/20 rounded-lg hover:border-white/40 transition-all duration-500 
                hover:scale-105 hover:shadow-md hover:shadow-purple-500/20 overflow-hidden text-xs">
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="relative flex items-center space-x-1">
                  <Maximize2 className="w-3 h-3 text-purple-400 group-hover:animate-pulse" />
                  <span className="font-medium text-white">Full Screen</span>
                </div>
              </button>
            </div>
          </div>

          {/* Search Results Section */}
          {showResults && (
            <div className="space-y-3 animate-fadeIn">
              <h2 className="text-sm font-bold text-white flex items-center space-x-2">
                <FileText className="w-4 h-4 text-green-200" />
                <span>Search Results</span>
                <div className="flex-1 h-px bg-gradient-to-r from-green-400/60 via-emerald-400/40 to-transparent"></div>
                <button 
                  onClick={() => setShowResults(false)}
                  className="p-0.5 rounded-full bg-white/10 hover:bg-white/20 transition-colors duration-300"
                >
                  <X className="w-3 h-3 text-gray-300 hover:text-white" />
                </button>
              </h2>
              
              <div className="relative backdrop-blur-2xl bg-gradient-to-br from-green-500/20 to-emerald-500/15 
                border border-emerald-400/40 rounded-xl p-3 shadow-lg shadow-emerald-500/30 overflow-hidden">
                
                <div className="absolute inset-0 bg-gradient-to-r from-green-500/10 to-emerald-500/10 opacity-0 hover:opacity-100 transition-opacity duration-500"></div>
                
                <div className="relative space-y-3">
                  {apiError ? (
                    <div className="text-center p-3 bg-red-500/20 border border-red-400/40 rounded-lg text-red-100 text-sm">
                      {apiError}
                    </div>
                  ) : searchResults.length > 0 ? (
                    <div className="space-y-2 max-h-60 overflow-y-auto custom-scrollbar pr-2">
                      {searchResults.map((result, index) => (
                        <div 
                          key={index} 
                          className="group relative p-2 bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-xl 
                            border border-white/20 rounded-lg hover:border-emerald-400/60 transition-all duration-300"
                        >
                          <div className="flex items-start space-x-2">
                            <div className="w-2 h-2 mt-1.5 bg-emerald-400 rounded-full animate-pulse flex-shrink-0"></div>
                            <div className="flex-1">
                              <p className="text-xs text-white/90">{result.content}</p>
                              <div className="flex items-center justify-between mt-1">
                                <span className="text-[10px] text-emerald-300 font-medium">{result.start_timestamp}</span>
                                <a 
                                  href={`${videoLink.split('&t=')[0].split('?t=')[0]}?t=${Math.floor(result.start_time)}s`} 
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[10px] bg-emerald-500/20 hover:bg-emerald-500/30 px-1.5 py-0.5 rounded 
                                    border border-emerald-400/40 text-emerald-100 hover:text-white transition-colors duration-300"
                                >
                                  Jump to
                                </a>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center p-3 bg-blue-500/20 border border-blue-400/40 rounded-lg text-blue-100 text-sm">
                      No results found for your search query.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer with Search Button */}
        <div className="relative p-4 pt-2 bg-gradient-to-t from-black/50 to-transparent backdrop-blur-xl border-t border-white/10 flex-shrink-0">
          <button
            onClick={handleSearch}
            disabled={!isVideoDetected || isSearching}
            className={`relative w-full font-bold py-3 px-6 rounded-xl transition-all duration-500 
              transform focus:outline-none focus:ring-1 focus:ring-blue-500/50 overflow-hidden group ${
              isVideoDetected && !isSearching
                ? 'bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 text-white hover:scale-[1.02] shadow-lg shadow-purple-500/40 hover:shadow-xl hover:shadow-purple-500/50'
                : 'bg-gradient-to-r from-gray-700 to-gray-600 text-gray-400 cursor-not-allowed'
            }`}
          >
            <div className="absolute inset-0 bg-gradient-to-r from-white/10 via-white/5 to-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            <div className="absolute inset-0 bg-gradient-to-45 from-blue-400/20 via-purple-400/20 to-pink-400/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
            
            {isSearching && (
              <>
                <div className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 rounded-full transition-all duration-500"
                  style={{ width: `${searchProgress}%` }}></div>
                <div className="absolute bottom-0 left-0 h-0.5 bg-gradient-to-r from-white/50 to-transparent rounded-full animate-pulse"
                  style={{ width: `${searchProgress}%` }}></div>
              </>
            )}
            
            <div className="relative flex items-center justify-center space-x-2 text-sm">
              {isSearching ? (
                <>
                  <div className="relative">
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                    <div className="absolute inset-0 w-5 h-5 border-2 border-transparent border-t-purple-400 rounded-full animate-spin" 
                      style={{animationDirection: 'reverse', animationDuration: '0.8s'}}></div>
                  </div>
                  <span className="font-bold">Processing</span>
                  <div className="flex space-x-0.5">
                    <div className="w-1.5 h-1.5 bg-white rounded-full animate-bounce"></div>
                    <div className="w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    <div className="w-1.5 h-1.5 bg-white rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                  </div>
                  <div className="text-xs font-medium opacity-80">
                    {searchProgress < 30 ? 'Initializing' : 
                     searchProgress < 60 ? 'Analyzing' : 
                     searchProgress < 90 ? 'Processing' : 'Finalizing'}
                  </div>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 group-hover:animate-pulse drop-shadow-md" />
                  <span className="font-bold">
                    {isVideoDetected ? 'Start Smart Seek' : 'No Video'}
                  </span>
                  <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform duration-500 drop-shadow-md" />
                </>
              )}
            </div>
            
            {isVideoDetected && !isSearching && (
              <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-80 transition-opacity duration-500"
                style={{
                  background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%)',
                  animation: 'shimmer 1s infinite'
                }}></div>
            )}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); opacity: 0.7; }
          50% { transform: translateY(-20px) rotate(180deg); opacity: 1; }
        }
        
        @keyframes movePattern {
          0% { transform: translateX(-50px); }
          100% { transform: translateX(50px); }
        }
        
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(5px); }
          to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
        
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(165, 180, 252, 0.5) transparent;
        }
        
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
          border-radius: 3px;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background-color: rgba(165, 180, 252, 0.5);
          border-radius: 3px;
          transition: background-color 0.3s;
        }
        
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background-color: rgba(165, 180, 252, 0.7);
        }
        
        .rounded-xl {
          border-radius: 12px;
        }
        
        @supports (scrollbar-color: auto) {
          .custom-scrollbar {
            scrollbar-width: thin;
            scrollbar-color: rgba(165, 180, 252, 0.5) transparent;
          }
        }
        ::-webkit-scrollbar {
          width: 0px;
          background: transparent;
        }
      `}</style>
    </div>
  );
}

export default App;