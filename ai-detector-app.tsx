import React, { useState } from 'react';
import { Upload, Image, Video, CheckCircle, XCircle, AlertCircle, Activity, FileText, Zap, BarChart3, Eye } from 'lucide-react';

const AIDetectorApp = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [fileType, setFileType] = useState(null);

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
    setAnalyzing(true);
    
    // Simulate analysis with realistic timing
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Simulate realistic hybrid analysis results
    const forensicScore = Math.random() * 0.4 + 0.3; // 0.3-0.7
    const mlScore = Math.random() * 0.4 + 0.4; // 0.4-0.8
    const metadataScore = Math.random() > 0.5 ? 0.8 : 0.2;
    
    // Weighted ensemble: 40% forensic, 40% ML, 20% metadata
    const finalScore = (0.4 * forensicScore) + (0.4 * mlScore) + (0.2 * metadataScore);
    const isAI = finalScore > 0.5;
    
    const mockResults = {
      prediction: isAI ? 'AI Generated' : 'Real',
      confidence: (finalScore * 100).toFixed(1),
      forensicAnalysis: {
        elaScore: (forensicScore * 100).toFixed(1),
        noisePattern: forensicScore > 0.5 ? 'Suspicious' : 'Natural',
        frequencyAnalysis: forensicScore > 0.5 ? 'Abnormal patterns detected' : 'Normal distribution',
        visualArtifacts: forensicScore > 0.6 ? 'Detected' : 'None found'
      },
      metadataAnalysis: {
        hasExif: metadataScore < 0.5,
        cameraModel: metadataScore < 0.5 ? 'Canon EOS 5D' : 'Not found',
        software: metadataScore > 0.5 ? 'AI Generation Tool' : 'Adobe Photoshop',
        suspicious: metadataScore > 0.5
      },
      mlAnalysis: {
        transferLearningScore: (mlScore * 100).toFixed(1),
        featureBasedScore: ((mlScore + 0.1) * 100).toFixed(1),
        confidence: mlScore > 0.5 ? 'High' : 'Medium'
      },
      votingSystem: {
        forensicVote: forensicScore > 0.5 ? 'AI' : 'Real',
        mlVote: mlScore > 0.5 ? 'AI' : 'Real',
        metadataVote: metadataScore > 0.5 ? 'AI' : 'Real',
        totalVotes: [forensicScore, mlScore, metadataScore].filter(s => s > 0.5).length
      },
      processingTime: fileType === 'video' ? '4.2s' : '1.8s'
    };
    
    setResults(mockResults);
    setAnalyzing(false);
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
                      <img src={preview} alt="Preview" className="w-full h-48 object-cover" />
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
                  <li>• Error Level Analysis (ELA)</li>
                  <li>• Noise Pattern Detection</li>
                  <li>• Frequency Domain Analysis</li>
                  <li>• Metadata Extraction</li>
                  <li>• Transfer Learning (ResNet)</li>
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
                    <span>Extracting metadata and EXIF data...</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Performing Error Level Analysis...</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Analyzing noise patterns and frequency domain...</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-blue-400 rounded-full mr-3 animate-pulse"></div>
                    <span>Running transfer learning model...</span>
                  </div>
                </div>
              </div>
            )}

            {results && (
              <div className="space-y-6">
                {/* Main Result */}
                <div className={`bg-gradient-to-r ${results.prediction === 'AI Generated' ? 'from-red-600/20 to-orange-600/20 border-red-500/50' : 'from-green-600/20 to-emerald-600/20 border-green-500/50'} backdrop-blur-md rounded-xl p-6 border`}>
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3">
                      {results.prediction === 'AI Generated' ? (
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
                        className={`h-2 rounded-full ${results.prediction === 'AI Generated' ? 'bg-red-500' : 'bg-green-500'}`}
                        style={{ width: `${results.confidence}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Voting System */}
                <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-purple-500/30">
                  <h3 className="text-lg font-semibold mb-4 flex items-center">
                    <BarChart3 className="w-5 h-5 mr-2" />
                    Voting System Results
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`w-16 h-16 mx-auto mb-2 rounded-full flex items-center justify-center ${results.votingSystem.forensicVote === 'AI' ? 'bg-red-600/30 border-2 border-red-500' : 'bg-green-600/30 border-2 border-green-500'}`}>
                        <FileText className="w-8 h-8" />
                      </div>
                      <p className="text-sm font-semibold">Forensic</p>
                      <p className="text-xs text-gray-400">{results.votingSystem.forensicVote}</p>
                    </div>
                    <div className="text-center">
                      <div className={`w-16 h-16 mx-auto mb-2 rounded-full flex items-center justify-center ${results.votingSystem.mlVote === 'AI' ? 'bg-red-600/30 border-2 border-red-500' : 'bg-green-600/30 border-2 border-green-500'}`}>
                        <Activity className="w-8 h-8" />
                      </div>
                      <p className="text-sm font-semibold">ML Model</p>
                      <p className="text-xs text-gray-400">{results.votingSystem.mlVote}</p>
                    </div>
                    <div className="text-center">
                      <div className={`w-16 h-16 mx-auto mb-2 rounded-full flex items-center justify-center ${results.votingSystem.metadataVote === 'AI' ? 'bg-red-600/30 border-2 border-red-500' : 'bg-green-600/30 border-2 border-green-500'}`}>
                        <FileText className="w-8 h-8" />
                      </div>
                      <p className="text-sm font-semibold">Metadata</p>
                      <p className="text-xs text-gray-400">{results.votingSystem.metadataVote}</p>
                    </div>
                  </div>
                  <div className="mt-4 text-center text-sm bg-purple-600/20 rounded-lg p-2">
                    <span className="font-semibold">{results.votingSystem.totalVotes}/3</span> modules voted for {results.prediction}
                  </div>
                </div>

                {/* Detailed Analysis */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Forensic Analysis */}
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-5 border border-purple-500/30">
                    <h4 className="font-semibold mb-3 flex items-center text-sm">
                      <FileText className="w-4 h-4 mr-2" />
                      Forensic Analysis
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">ELA Score:</span>
                        <span className="font-semibold">{results.forensicAnalysis.elaScore}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Noise Pattern:</span>
                        <span className={`font-semibold ${results.forensicAnalysis.noisePattern === 'Suspicious' ? 'text-red-400' : 'text-green-400'}`}>
                          {results.forensicAnalysis.noisePattern}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Frequency Analysis:</span>
                        <span className="text-xs">{results.forensicAnalysis.frequencyAnalysis}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Visual Artifacts:</span>
                        <span className={`font-semibold ${results.forensicAnalysis.visualArtifacts === 'Detected' ? 'text-red-400' : 'text-green-400'}`}>
                          {results.forensicAnalysis.visualArtifacts}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Metadata Analysis */}
                  <div className="bg-white/10 backdrop-blur-md rounded-xl p-5 border border-purple-500/30">
                    <h4 className="font-semibold mb-3 flex items-center text-sm">
                      <FileText className="w-4 h-4 mr-2" />
                      Metadata Analysis
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">EXIF Data:</span>
                        <span className={`font-semibold ${results.metadataAnalysis.hasExif ? 'text-green-400' : 'text-red-400'}`}>
                          {results.metadataAnalysis.hasExif ? 'Present' : 'Missing'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Camera Model:</span>
                        <span className="text-xs">{results.metadataAnalysis.cameraModel}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Software:</span>
                        <span className="text-xs">{results.metadataAnalysis.software}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Suspicious:</span>
                        <span className={`font-semibold ${results.metadataAnalysis.suspicious ? 'text-red-400' : 'text-green-400'}`}>
                          {results.metadataAnalysis.suspicious ? 'Yes' : 'No'}
                        </span>
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
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIDetectorApp;