import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-primary-100">
      <div className="relative isolate">
        {/* Background Effects */}
        <div className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80">
          <div className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-primary-200 to-primary-400 opacity-20 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]" />
        </div>

        {/* Main Content */}
        <div className="mx-auto max-w-7xl px-6 py-24 sm:py-32 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl bg-gradient-to-r from-primary-600 to-primary-800 bg-clip-text text-transparent animate-fade-in">
              Welcome to LumiDoc AI Chat
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600 max-w-2xl mx-auto">
              Experience intelligent conversations powered by advanced AI. Connect, learn, and explore with our cutting-edge chat application.
            </p>
            <div className="mt-10 flex items-center justify-center gap-6">
              <Link
                to="/login"
                className="rounded-xl bg-primary-600 px-8 py-3.5 text-sm font-semibold text-white shadow-sm hover:bg-primary-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary-600 transition-all duration-300 hover:scale-105"
              >
                Login
              </Link>
              <Link
                to="/signup"
                className="rounded-xl bg-white px-8 py-3.5 text-sm font-semibold text-primary-600 shadow-sm ring-1 ring-inset ring-primary-200 hover:ring-primary-300 hover:bg-primary-50 transition-all duration-300 hover:scale-105"
              >
                Sign Up
              </Link>
            </div>
          </div>
        </div>

        {/* Features Section */}
        <div className="mx-auto max-w-7xl px-6 lg:px-8 pb-24">
          <div className="mx-auto grid max-w-2xl grid-cols-1 gap-x-8 gap-y-8 sm:gap-y-10 lg:mx-0 lg:max-w-none lg:grid-cols-3">
            {/* Feature 1 */}
            <div className="group relative bg-white rounded-3xl shadow-xl p-8 hover:scale-105 transition-all duration-300">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-100 group-hover:bg-primary-50 transition-colors duration-300">
                <svg className="h-7 w-7 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h3 className="mt-6 text-xl font-semibold leading-7 text-gray-900">Smart Conversations</h3>
              <p className="mt-2 text-base leading-7 text-gray-600">Engage in intelligent discussions with our advanced AI system powered by cutting-edge technology.</p>
            </div>

            {/* Feature 2 */}
            <div className="group relative bg-white rounded-3xl shadow-xl p-8 hover:scale-105 transition-all duration-300">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-100 group-hover:bg-primary-50 transition-colors duration-300">
                <svg className="h-7 w-7 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
              </div>
              <h3 className="mt-6 text-xl font-semibold leading-7 text-gray-900">Customizable Experience</h3>
              <p className="mt-2 text-base leading-7 text-gray-600">Tailor your chat experience with personalized settings and preferences.</p>
            </div>

            {/* Feature 3 */}
            <div className="group relative bg-white rounded-3xl shadow-xl p-8 hover:scale-105 transition-all duration-300">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary-100 group-hover:bg-primary-50 transition-colors duration-300">
                <svg className="h-7 w-7 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="mt-6 text-xl font-semibold leading-7 text-gray-900">Secure & Private</h3>
              <p className="mt-2 text-base leading-7 text-gray-600">Your conversations are protected with enterprise-grade security and encryption.</p>
            </div>
          </div>
        </div>

        {/* Bottom Gradient */}
        <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]">
          <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-primary-300 to-primary-500 opacity-20 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
        </div>
      </div>
    </div>
  );
};

export default Home; 