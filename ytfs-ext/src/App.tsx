import  { useState } from 'react';

function App() {
  const [videoLink, setVideoLink] = useState('');
  const [searchKeyword, setSearchKeyword] = useState('');
  const [semanticSearch, setSemanticSearch] = useState(false);
  const [fullTextSearch, setFullTextSearch] = useState(false);
  const [advancedSearch, setAdvancedSearch] = useState(false);

  const handleVideoLinkChange = (e:any) => {
    setVideoLink(e.target.value);
  };

  const handleSearchKeywordChange = (e:any) => {
    setSearchKeyword(e.target.value);
  };

  const handleSearch = () => {
    if (!videoLink.trim()) {
      alert('Please provide a video link first!');
      return;
    }
    if (!searchKeyword.trim()) {
      alert('Please enter a search keyword!');
      return;
    }

    const activeSearchTypes = [];
    if (semanticSearch) activeSearchTypes.push('Semantic Search');
    if (fullTextSearch) activeSearchTypes.push('Full Text Search');
    if (advancedSearch) activeSearchTypes.push('Advanced Search');

    if (activeSearchTypes.length === 0) {
      alert('Please select at least one search method!');
      return;
    }

    console.log('Searching video:', videoLink);
    console.log('Keyword:', searchKeyword);
    console.log('Active search types:', activeSearchTypes);
    
    // Here you would implement the actual search functionality
    alert(`Searching for "${searchKeyword}" using: ${activeSearchTypes.join(', ')}`);
  };

  return (
    <div className="min-w-80 max-w-md mx-auto bg-gray-900 shadow-2xl rounded-lg overflow-hidden border border-gray-700">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-700 px-6 py-4">
        <h1 className="text-xl font-bold text-white text-center">
          Video Search Extension
        </h1>
      </div>

      <div className="p-6 space-y-6">
        {/* Video Link Input */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Video Link
          </label>
          <div className="relative">
            <input
              type="url"
              value={videoLink}
              onChange={handleVideoLinkChange}
              placeholder="https://youtube.com/watch?v=..."
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent placeholder-gray-500 text-white"
            />
            <div className="absolute inset-y-0 right-0 flex items-center pr-3">
              <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
            </div>
          </div>
        </div>

        {/* Search Keyword Input */}
      
          <div className="relative">
            <input
              type="text"
              value={searchKeyword}
              onChange={handleSearchKeywordChange}
              placeholder="Enter keyword to search in video..."
              className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent placeholder-gray-500 text-white"
            />
            <div className="absolute inset-y-0 right-0 flex items-center pr-3">
              <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Search Method Toggles */}
        <div className="space-y-3">
          <label className="block text-sm font-medium text-gray-300">
            Search Methods
          </label>
          
          <div className="space-y-2">
            {/* Semantic Search Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-purple-900 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <span className="text-sm font-medium text-gray-200">Semantic Search</span>
              </div>
              <button
                onClick={() => setSemanticSearch(!semanticSearch)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  semanticSearch ? 'bg-indigo-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    semanticSearch ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {/* Full Text Search Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-green-900 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <span className="text-sm font-medium text-gray-200">Full Text Search</span>
              </div>
              <button
                onClick={() => setFullTextSearch(!fullTextSearch)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  fullTextSearch ? 'bg-indigo-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    fullTextSearch ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {/* Advanced Search Toggle */}
            <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg hover:bg-gray-750 transition-colors border border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-orange-900 rounded-full flex items-center justify-center">
                  <svg className="w-4 h-4 text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                  </svg>
                </div>
                <span className="text-sm font-medium text-gray-200">Advanced Search</span>
              </div>
              <button
                onClick={() => setAdvancedSearch(!advancedSearch)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  advancedSearch ? 'bg-indigo-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    advancedSearch ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Search Button */}
        <button
          onClick={handleSearch}
          className="w-full bg-gradient-to-r from-indigo-600 to-purple-700 hover:from-indigo-700 hover:to-purple-800 text-white font-medium py-3 px-4 rounded-lg transition-all duration-200 transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-gray-900"
        >
          <div className="flex items-center justify-center space-x-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <span>Start Search</span>
          </div>
        </button>

        {/* Status Indicator */}
        {(semanticSearch || fullTextSearch || advancedSearch) && (
          <div className="mt-4 p-3 bg-indigo-900 border border-indigo-700 rounded-lg">
            <div className="text-sm text-indigo-200">
              <strong>Active Methods:</strong>{' '}
              {[
                semanticSearch && 'Semantic',
                fullTextSearch && 'Full Text',
                advancedSearch && 'Advanced'
              ].filter(Boolean).join(', ')}
            </div>
          </div>
        )}
      </div>
   
  );
}

export default App;