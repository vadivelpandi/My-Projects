import React, { useState } from 'react';
import { Upload, Image, Video, CheckCircle, XCircle, AlertCircle, Activity, FileText, Zap, BarChart3, Eye, Grid } from 'lucide-react';

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [fileType, setFileType] = useState(null);

  // Auth State
  const [isMetadataUnlocked, setIsMetadataUnlocked] = useState(false);
  const [showMetadataAuth, setShowMetadataAuth] = useState(false);
  const [email, setEmail] = useState("");
  const [otp, setOtp] = useState("");
  const [otpSent, setOtpSent] = useState(false);
  const [loadingAuth, setLoadingAuth] = useState(false);

  const handleRequestOtp = async () => {
    setLoadingAuth(true);
    try {
      const res = await fetch('/auth/request-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      if (res.ok) {
        setOtpSent(true);
      } else {
        alert("Failed to send OTP. Check backend logs.");
      }
    } catch (e) { console.error(e); alert("Error sending OTP"); }
    setLoadingAuth(false);
  };

  const handleVerifyOtp = async () => {
    setLoadingAuth(true);
    try {
      const res = await fetch('/auth/verify-otp', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, code: otp })
      });
      if (res.ok) {
        setIsMetadataUnlocked(true);
        setShowMetadataAuth(false);
      } else {
        alert("Invalid Code");
      }
    } catch (e) { console.error(e); alert("Verification failed"); }
    setLoadingAuth(false);
  };

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(uploadedFile);

      if (uploadedFile.type.startsWith('image/')) {
        setFileType('image');
      } else if (uploadedFile.type.startsWith('video/')) {
        setFileType('video');
      }
      setResults(null);
    }
  };

  const analyzeContent = async () => {
    if (!file) return;

    setAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Failed to analyze content. Please ensure backend is running.");
    } finally {
      setAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setFileType(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <header className="border-b border-purple-500/30 bg-black/20 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-purple-600 p-2 rounded-lg">
                <Eye className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold">AI Content Detector</h1>
                <p className="text-sm text-purple-300">Hybrid Forensic & ML Analysis</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm">
              <Activity className="w-4 h-4 text-green-400" />
              <span className="text-green-400">System Active</span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Upload Content
              </h2>

              {!preview ? (
                <label className="block">
                  <div className="border-2 border-dashed border-purple-500/50 rounded-lg p-8 text-center cursor-pointer hover:border-purple-400 transition-colors">
                    <Upload className="w-12 h-12 mx-auto mb-3 text-purple-400" />
                    <p className="text-sm mb-2">Click to upload or drag and drop</p>
                    <p className="text-xs text-gray-400">Images (JPG, PNG) or Videos (MP4, MOV)</p>
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    accept="image/*,video/*"
                    onChange={handleFileUpload}
                  />
                </label>
              ) : (
                <div className="space-y-4">
                  <div className="relative rounded-lg overflow-hidden border border-purple-500/30">
                    {fileType === 'image' ? (
                      <img src={preview} alt="Preview" className="w-auto h-auto max-h-[500px] mx-auto object-contain" />
                    ) : (
                      <video src={preview} className="w-full h-48 object-cover" controls />
                    )}
                  </div>

                  <div className="flex items-center space-x-2 text-sm">
                    {fileType === 'image' ? <Image className="w-4 h-4" /> : <Video className="w-4 h-4" />}
                    <span className="truncate">{file?.name}</span>
                  </div>

                  {!results && (
                    <button
                      onClick={analyzeContent}
                      disabled={analyzing}
                      className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-purple-800 text-white py-3 rounded-lg font-semibold transition-colors flex items-center justify-center"
                    >
                      {analyzing ? (
                        <>
                          <Activity className="w-5 h-5 mr-2 animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <Zap className="w-5 h-5 mr-2" />
                          Analyze Content
                        </>
                      )}
                    </button>
                  )}

                  {results && (
                    <button
                      onClick={resetAnalysis}
                      className="w-full bg-slate-700 hover:bg-slate-600 text-white py-3 rounded-lg font-semibold transition-colors"
                    >
                      Upload New File
                    </button>
                  )}
                </div>
              )}
            </div>

            {/* Info Cards */}
            <div className="mt-6 space-y-3">
              <div className="bg-purple-600/20 backdrop-blur-md rounded-lg p-4 border border-purple-500/30">
                <h3 className="font-semibold mb-2 flex items-center text-sm">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Detection Methods
                </h3>
                <ul className="text-xs space-y-1 text-gray-300">
                  <li>• Multi-Model Ensemble (11 Models)</li>
                  <li>• Forensic Signal Analysis (ELA, Noise)</li>
                  <li>• CLIP Semantic Drift Check</li>
                  <li>• Metadata & Exif Validation</li>

                </ul>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {!results && !analyzing && (
              <div className="bg-white/5 backdrop-blur-md rounded-xl p-12 border border-purple-500/30 text-center">
                <AlertCircle className="w-16 h-16 mx-auto mb-4 text-purple-400 opacity-50" />
                <h3 className="text-xl font-semibold mb-2">Ready to Analyze</h3>
                <p className="text-gray-400">Upload an image or video to detect if it's AI-generated or real</p>
              </div>
            )}

            {analyzing && (
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-12 border border-purple-500/30 text-center">
                <Activity className="w-16 h-16 mx-auto mb-4 text-purple-400 animate-spin" />
                <h3 className="text-xl font-semibold mb-2">Analyzing Content</h3>
                <p className="text-gray-400 mb-6">Running hybrid forensic and ML analysis...</p>
                <div className="max-w-md mx-auto space-y-2 text-left text-sm">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Initializing Forensic & ML Engines...</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Running Deep Learning Ensemble...</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Performing Metadata & Signal Analysis...</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-purple-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Synthesizing Final Verdict...</span>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message Display */}
            {results && results.error && (
              <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-6 text-center text-white mb-6 animate-fadeIn">
                <h3 className="text-xl font-bold mb-2">Analysis Error</h3>
                <p className="font-mono text-sm">{results.error}</p>
                {results.traceback && (
                  <details className="mt-4 text-left">
                    <summary className="cursor-pointer text-red-300 text-xs text-left">View Traceback</summary>
                    <pre className="mt-2 text-[10px] bg-black/50 p-4 rounded overflow-x-auto text-left">
                      {results.traceback}
                    </pre>
                  </details>
                )}
              </div>
            )}

            {results && !results.error && (
              <div className="space-y-6">

                {/* Main Result */}
                <div className={`bg-gradient-to-r ${results.prediction !== 'Real / Authentic' ? 'from-red-600/20 to-orange-600/20 border-red-500/50' : 'from-green-600/20 to-emerald-600/20 border-green-500/50'} backdrop-blur-md rounded-xl p-6 border`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {results.prediction !== 'Real / Authentic' ? (
                        <XCircle className="w-10 h-10 text-red-400" />
                      ) : (
                        <CheckCircle className="w-10 h-10 text-green-400" />
                      )}
                      <div>
                        <h3 className="text-2xl font-bold">{results.prediction}</h3>
                        <p className="text-sm text-gray-300">Confidence: {results.confidence}%</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-gray-400">Processing Time</p>
                      <p className="text-lg font-semibold">{results.processingTime}</p>
                    </div>
                  </div>

                  <div className="bg-black/30 rounded-lg p-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span>Overall Confidence</span>
                      <span>{results.confidence}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${results.prediction !== 'Real / Authentic' ? 'bg-red-500' : 'bg-green-500'}`}
                        style={{ width: `${results.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Scoring Breakdown Card (New) */}
                {results.score_breakdown && (
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <BarChart3 className="w-5 h-5 mr-2" />
                      Scoring Breakdown
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* ML Score */}
                      <div className="bg-black/20 p-4 rounded-lg border border-purple-500/20 text-center">
                        <p className="text-gray-400 text-xs uppercase mb-1">ML Consensus</p>
                        <p className="text-2xl font-bold text-white mb-1">{results.score_breakdown.ml_confidence}%</p>
                        <span className={`text-xs px-2 py-0.5 rounded font-bold ${results.score_breakdown.ml_confidence > 50 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                          {results.score_breakdown.ml_confidence > 50 ? "AI Generated" : "Real / Authentic"}
                        </span>
                        <p className="text-xs text-blue-300 mt-2">Based on 11 Models</p>
                      </div>

                      {/* Forensic Score */}
                      <div className="bg-black/20 p-4 rounded-lg border border-purple-500/20 text-center">
                        <p className="text-gray-400 text-xs uppercase mb-1">Forensic Likelihood</p>
                        <p className={`text-2xl font-bold mb-1 ${results.score_breakdown.forensic_confidence > 50 ? 'text-white' : 'text-white'}`}>
                          {results.score_breakdown.forensic_confidence}%
                        </p>
                        <span className={`text-xs px-2 py-0.5 rounded font-bold ${results.score_breakdown.forensic_confidence > 50 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                          {results.score_breakdown.forensic_confidence > 50 ? "High Artifacts" : "Natural Signals"}
                        </span>
                        <p className="text-xs text-blue-300 mt-2">Signal Analysis</p>
                      </div>

                      {/* Fusion Reason */}
                      <div className="bg-black/20 p-4 rounded-lg border border-purple-500/20 text-center flex flex-col justify-center">
                        <p className="text-gray-400 text-xs uppercase mb-1">Fusion Logic</p>
                        <p className="text-lg font-bold text-purple-300">
                          {results.score_breakdown.fusion_reason}
                        </p>
                      </div>
                    </div>
                  </div>
                )}



                {/* Global vs Local Conflict Analysis Card (New) */}
                {results.detailed_steps && results.detailed_steps.step4_patches && (
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30">
                    <h3 className="text-lg font-semibold mb-4 flex items-center">
                      <Grid className="w-5 h-5 mr-2" />
                      Global vs Local Conflict Analysis
                    </h3>

                    {/* Metrics Row */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                      <div className="bg-black/20 p-3 rounded-lg border border-purple-500/20">
                        <p className="text-gray-400 text-xs uppercase mb-1">Conflict Status</p>
                        <p className={`font-bold ${results.conflict_detected === 'Yes' ? 'text-red-400' : 'text-green-400'}`}>
                          {results.conflict_detected === 'Yes' ? 'Conflict Detected' : 'Consistent'}
                        </p>
                      </div>
                      <div className="bg-black/20 p-3 rounded-lg border border-purple-500/20">
                        <p className="text-gray-400 text-xs uppercase mb-1">Consistency Level</p>
                        <p className={`font-bold ${results.patch_consistency === 'High' ? 'text-green-400' : results.patch_consistency === 'Low' ? 'text-red-400' : 'text-yellow-400'}`}>
                          {results.patch_consistency}
                        </p>
                      </div>
                      <div className="bg-black/20 p-3 rounded-lg border border-purple-500/20">
                        <p className="text-gray-400 text-xs uppercase mb-1">Patch Variance</p>
                        <p className="text-white font-bold">{results.detailed_steps.step4_patches.variance}</p>
                      </div>
                    </div>

                    <div className="flex flex-col md:flex-row gap-6">
                      {/* 4x4 Grid Visualization with Image Overlay */}
                      <div className="flex-shrink-0">
                        <p className="text-gray-400 text-xs mb-2 text-center">AI Spatial Conflict Heatmap</p>
                        <div className="relative w-48 h-48 mx-auto rounded-lg overflow-hidden border-2 border-purple-500/50 shadow-lg group">
                          {/* Background Image */}
                          <img
                            src={preview}
                            alt="Analysis Preview"
                            className="absolute inset-0 w-full h-full object-cover opacity-60 group-hover:opacity-80 transition-opacity"
                          />

                          {/* Grid Overlay */}
                          <div className="absolute inset-0 grid grid-cols-4 grid-rows-4 gap-0.5 p-0.5">
                            {results.detailed_steps.step4_patches.patch_scores.map((patch) => (
                              <div
                                key={patch.id}
                                className={`relative flex items-center justify-center transition-all duration-300 hover:scale-[1.02] hover:z-10 cursor-help
                                  ${patch.verdict === 'AI Generated'
                                    ? 'bg-red-500/40 border border-red-500/30'
                                    : 'bg-green-500/10 border border-green-500/20'}`}
                                title={`Patch ${patch.row},${patch.col}: ${patch.verdict} (${patch.confidence}%)`}
                              >
                                {patch.verdict === 'AI Generated' && patch.confidence > 70 && (
                                  <AlertCircle className="w-4 h-4 text-red-100 drop-shadow-[0_0_8px_rgba(239,68,68,1)] animate-pulse" />
                                )}

                                {/* Hover Indicator */}
                                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 hover:bg-white/10 transition-opacity" />
                              </div>
                            ))}
                          </div>

                          {/* Corner Indicators */}
                          <div className="absolute top-0 left-0 p-1">
                            <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-ping" />
                          </div>
                        </div>
                      </div>

                      {/* Details & Suspected Regions */}
                      <div className="flex-1 space-y-4">
                        <div>
                          <p className="text-gray-400 text-xs uppercase mb-1">Suspected Regions / Anomalies</p>
                          <div className="bg-black/20 p-3 rounded-lg border border-purple-500/20 h-full">
                            <p className="text-sm text-gray-200">
                              {results.suspected_regions}
                            </p>
                            {results.conflict_detected === 'Yes' && (
                              <p className="text-xs text-red-300 mt-2 flex items-center">
                                <AlertCircle className="w-3 h-3 mr-1" />
                                Disagreement between Global Analysis and Local Patches suggests partial manipulation.
                              </p>
                            )}
                          </div>
                        </div>

                        <div>
                          <p className="text-gray-400 text-xs uppercase mb-1">Grid Stats</p>
                          <div className="bg-black/20 p-3 rounded-lg border border-purple-500/20 flex justify-between">
                            <span className="text-xs text-red-300">AI Patches: {results.detailed_steps.step4_patches.ai_patch_ratio.split('/')[0]}</span>
                            <span className="text-xs text-gray-400">Total: 16</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Model Consensus */}
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30">
                  <h3 className="text-lg font-semibold mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Model Consensus ({results.modelConsensus.totalModels} Models)
                  </h3>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="text-center bg-white/5 rounded-lg p-3">
                      <span className="block text-2xl font-bold text-green-400">{results.modelConsensus.realVotes}</span>
                      <span className="text-sm text-gray-400">Votes for Real</span>
                    </div>
                    <div className="text-center bg-white/5 rounded-lg p-3">
                      <span className="block text-2xl font-bold text-red-400">{results.modelConsensus.aiVotes}</span>
                      <span className="text-sm text-gray-400">Votes for AI</span>
                    </div>
                  </div>
                  <div className="text-center text-sm">
                    <p className="text-gray-300">Agreement Level: <span className="font-bold">{results.modelConsensus.agreement}%</span></p>
                  </div>
                </div>

                {/* Detailed Models Breakdown */}
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30 lg:col-span-2">
                  <h3 className="text-lg font-semibold mb-4 text-white">Detailed Model Breakdown</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-3">
                    {results.detailedModels && results.detailedModels.map((model, idx) => (
                      <div key={idx} className="flex items-center justify-between bg-black/20 p-3 rounded-lg border border-purple-500/10">
                        <span className="text-xs text-gray-300 truncate w-1/2" title={model.model}>{model.model.split('/')[1] || model.model}</span>
                        <div className="flex items-center space-x-3">
                          <span className="text-xs text-gray-400">{model.confidence}% Conf</span>
                          <span className={`text-xs px-2 py-1 rounded font-bold ${model.verdict === 'AI' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                            {model.verdict}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Forensic Analysis */}
                  {/* 10-Step Forensic Report */}
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-5 border border-purple-500/30">
                    <h4 className="font-semibold mb-3 flex items-center text-sm">
                      <Zap className="w-4 h-4 mr-2" />
                      Multi-Layer Forensic Analysis
                    </h4>
                    <div className="space-y-4 text-sm">

                      {/* Scale Consistency */}
                      <div>
                        <p className="text-gray-400 text-xs mb-1">Multi-Scale Consistency (Steps 1 & 3)</p>
                        <div className="flex justify-between items-center bg-black/20 p-2 rounded">
                          <span>Stability</span>
                          <span className={results.detailed_steps.step3_multiscale.status.includes('Unstable') ? 'text-red-400 font-bold' : 'text-green-400 font-bold'}>
                            {results.detailed_steps.step3_multiscale.status}
                          </span>
                        </div>
                      </div>



                      {/* Frequency & Camera */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className="bg-black/20 p-2 rounded">
                          <p className="text-gray-400 text-xs">FFT Analysis</p>
                          <span className={results.detailed_steps.step5_frequency?.fft_verdict !== 'Natural' ? 'text-red-400 font-bold' : 'text-green-400 font-bold'}>
                            {results.detailed_steps.step5_frequency?.fft_verdict || 'N/A'}
                          </span>
                        </div>
                        <div className="bg-black/20 p-2 rounded">
                          <p className="text-gray-400 text-xs">Camera Noise</p>
                          <span className={results.detailed_steps.step2_camera?.prnu_status !== 'Normal' ? 'text-red-400 font-bold' : 'text-green-400 font-bold'}>
                            {results.detailed_steps.step2_camera?.prnu_status || 'N/A'}
                          </span>
                        </div>
                      </div>

                      {/* Semantic Drift */}
                      <div>
                        <p className="text-gray-400 text-xs mb-1">Semantic Drift (CLIP) (Step 8)</p>
                        <div className="flex justify-between items-center bg-black/20 p-2 rounded">
                          <span>AI Semantics Score</span>
                          <span className="font-bold text-blue-300">
                            {results.detailed_steps.step8_clip.clip_ai_score}%
                          </span>
                        </div>
                        <p className="text-xs text-gray-500 mt-1">{results.detailed_steps.step8_clip.verdict}</p>
                      </div>

                    </div>
                  </div>

                  {/* Physics & Color */}
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-5 border border-purple-500/30">
                    <h4 className="font-semibold mb-3 flex items-center text-sm">
                      <Eye className="w-4 h-4 mr-2" />
                      Physics & Color Rules
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between border-b border-gray-700 pb-2">
                        <span className="text-gray-400">Color Saturation</span>
                        <span className="font-bold">{results.detailed_steps.step6_color?.sat_verdict || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between border-b border-gray-700 pb-2">
                        <span className="text-gray-400">Lighting Logic</span>
                        <span className="font-bold">{results.detailed_steps.step7_physics?.lighting_physics || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between border-b border-gray-700 pb-2">
                        <span className="text-gray-400">Edge Integrity</span>
                        <span className="font-bold">{results.detailed_steps.step9_structure?.integrity || 'N/A'}</span>
                      </div>
                      <div className="flex justify-between pt-2">
                        <span className="text-gray-400">DWT High-Freq</span>
                        <span className="font-bold">{results.detailed_steps.step5_frequency?.dwt_verdict || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ML Analysis */}
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-5 border border-purple-500/30">
                  <h4 className="font-semibold mb-3 flex items-center">
                    <Activity className="w-5 h-5 mr-2" />
                    Machine Learning Analysis
                  </h4>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400 mb-1">Transfer Learning</p>
                      <p className="text-2xl font-bold">{results.mlAnalysis.transferLearningScore}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400 mb-1">Feature-Based</p>
                      <p className="text-2xl font-bold">{results.mlAnalysis.featureBasedScore}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400 mb-1">Confidence Level</p>
                      <p className="text-2xl font-bold">{results.mlAnalysis.confidence}</p>
                    </div>
                  </div>
                </div>
              </div>
            )}





            {/* Raw Metadata Details Cards (Protected) */}
            {results && results.metadata_report && results.metadata_report.debug_raw_tags && (() => {
              if (!showMetadataAuth && !isMetadataUnlocked) {
                return (
                  <div className="lg:col-span-2 mt-6">
                    <div className="bg-white/10 backdrop-blur-md rounded-xl p-8 border border-purple-500/30 text-center">
                      <FileText className="w-12 h-12 mx-auto mb-4 text-purple-400 opacity-50" />
                      <h3 className="text-xl font-semibold mb-2 text-white">Raw Metadata Hidden</h3>
                      <p className="text-gray-400 mb-6 text-sm">To view sensitive raw metadata, verification is required.</p>
                      <button
                        onClick={() => setShowMetadataAuth(true)}
                        className="bg-purple-600 hover:bg-purple-500 text-white px-6 py-2 rounded-lg font-semibold transition-colors"
                      >
                        View Metadata
                      </button>
                    </div>
                  </div>
                );
              }

              if (showMetadataAuth && !isMetadataUnlocked) {
                return (
                  <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
                    <div className="bg-slate-900 border border-purple-500/50 rounded-xl p-8 max-w-md w-full mx-4 shadow-2xl">
                      <div className="flex justify-between items-center mb-6">
                        <h3 className="text-xl font-semibold text-white">Metadata Access Verification</h3>
                        <button onClick={() => setShowMetadataAuth(false)}><XCircle className="w-6 h-6 text-gray-400 hover:text-white" /></button>
                      </div>

                      {!otpSent ? (
                        <div className="space-y-4">
                          <p className="text-gray-300 text-sm">Please enter your Gmail address to receive a verification code.</p>
                          <input
                            type="email"
                            placeholder="Enter your Gmail"
                            className="w-full bg-black/30 border border-purple-500/30 rounded-lg p-3 text-white focus:outline-none focus:border-purple-400"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                          />
                          <button
                            onClick={handleRequestOtp}
                            disabled={loadingAuth || !email}
                            className="w-full bg-purple-600 hover:bg-purple-500 disabled:bg-purple-800 disabled:cursor-not-allowed text-white py-3 rounded-lg font-bold transition-colors"
                          >
                            {loadingAuth ? "Sending Code..." : "Send Verification Code"}
                          </button>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2 text-green-400 bg-green-900/20 p-3 rounded-lg mb-4">
                            <CheckCircle className="w-5 h-5" />
                            <span className="text-sm">Code sent to {email}</span>
                          </div>
                          <p className="text-gray-300 text-sm">Enter the 4-digit code sent to your email.</p>
                          <input
                            type="text"
                            placeholder="0000"
                            maxLength="4"
                            className="w-full bg-black/30 border border-purple-500/30 rounded-lg p-3 text-white text-center text-2xl tracking-widest focus:outline-none focus:border-purple-400"
                            value={otp}
                            onChange={(e) => setOtp(e.target.value)}
                          />
                          <button
                            onClick={handleVerifyOtp}
                            disabled={loadingAuth || otp.length !== 4}
                            className="w-full bg-green-600 hover:bg-green-500 disabled:bg-green-800 disabled:cursor-not-allowed text-white py-3 rounded-lg font-bold transition-colors"
                          >
                            {loadingAuth ? "Verifying..." : "Verify & Unlock"}
                          </button>
                          <button
                            onClick={() => setOtpSent(false)}
                            className="w-full text-gray-400 hover:text-white text-sm py-2"
                          >
                            Change Email
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                );
              }

              // Group tags by prefix (e.g. "File:FileSize" -> Group "File")
              const groups = {};
              const rawTags = results.metadata_report.debug_raw_tags;
              Object.keys(rawTags).forEach(key => {
                const parts = key.split(':');
                const groupName = parts.length > 1 ? parts[0] : 'General';
                if (!groups[groupName]) groups[groupName] = {};
                groups[groupName][key] = rawTags[key];
              });

              return (
                <div className="lg:col-span-2 mt-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-xl font-semibold text-white flex items-center">
                      <FileText className="w-6 h-6 mr-2" />
                      Raw Metadata Dump
                    </h3>
                    <button onClick={() => setIsMetadataUnlocked(false)} className="text-xs text-red-300 hover:text-red-200 border border-red-500/30 px-3 py-1 rounded">Lock Metadata</button>
                  </div>

                  <div className="space-y-4">
                    {Object.keys(groups).sort().map(groupName => (
                      <div key={groupName} className="bg-white/10 backdrop-blur-md rounded-xl p-4 border border-purple-500/30">
                        <h4 className="font-bold text-purple-300 mb-2 border-b border-purple-500/20 pb-1">
                          {groupName} ({Object.keys(groups[groupName]).length})
                        </h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 text-xs font-mono max-h-60 overflow-y-auto custom-scrollbar">
                          {Object.entries(groups[groupName]).map(([k, v]) => (
                            <div key={k} className="flex flex-col mb-1 hover:bg-white/5 p-1 rounded">
                              <span className="text-gray-400 font-semibold truncate" title={k}>{k.split(':').slice(1).join(':') || k}</span>
                              <span className="text-white break-all">{v}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* Video Frame Analysis Section */}

            {results && results.type === 'video' && results.videoAnalysis && (
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30 lg:col-span-2">
                <h3 className="text-lg font-semibold mb-4 text-white flex items-center">
                  <Video className="w-5 h-5 mr-2" />
                  Frame-by-Frame Analysis (5 Sampled Frames)
                </h3>

                <div className="space-y-6">
                  {results.videoAnalysis.map((frame, frameIdx) => (
                    <div key={frameIdx} className="bg-black/20 rounded-xl p-4 border border-purple-500/20">
                      <div className="flex flex-col md:flex-row gap-4">
                        {/* Frame Image */}
                        <div className="md:w-1/3">
                          <div className="relative rounded-lg overflow-hidden border border-gray-700 aspect-video group">
                            {/* Base Image */}
                            <img
                              src={frame.imageBase64}
                              alt={`Frame ${frame.frameIndex}`}
                              className="w-full h-full object-cover"
                            />



                            <div className="absolute top-2 left-2 bg-black/70 px-2 py-1 rounded text-xs text-white z-10">
                              Frame {frame.frameIndex}
                            </div>
                            <div className={`absolute bottom-2 right-2 px-2 py-1 rounded text-xs font-bold z-10 ${frame.consensus === 'AI Generated' ? 'bg-red-500 text-white' : 'bg-green-500 text-white'}`}>
                              {frame.consensus}
                            </div>
                          </div>
                        </div>

                        {/* Frame Details */}
                        <div className="md:w-2/3">
                          <div className="mb-3 flex justify-between items-center">
                            <h4 className="font-semibold text-gray-200">Ensemble Verdict: <span className={frame.consensus === 'AI Generated' ? 'text-red-400' : 'text-green-400'}>{frame.consensus}</span></h4>
                            <span className="text-xs text-gray-400">Confidence: {frame.confidence}%</span>
                          </div>

                          <p className="text-xs text-gray-400 mb-2">Why? Individual Model Breakdown:</p>
                          <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto pr-2 custom-scrollbar">
                            {frame.modelBreakdown.filter(m => m.status !== 'Deactivated').map((model, mIdx) => (
                              <div key={mIdx} className={`text-xs p-2 rounded flex justify-between items-center border ${model.verdict === 'AI' ? 'bg-red-900/10 border-red-500/20' : 'bg-green-900/10 border-green-500/20'}`}>
                                <span className="truncate w-2/3" title={model.model}>{model.model.split('/')[1] || model.model}</span>
                                <span className={`font-bold ${model.verdict === 'AI' ? 'text-red-400' : 'text-green-400'}`}>{model.verdict}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        </div>
      </div >
    </div >
  );
};

export default App;
