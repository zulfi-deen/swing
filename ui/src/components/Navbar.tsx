import { Link, useLocation } from 'react-router-dom';
import { TrendingUp, List, Cpu, BarChart3 } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: TrendingUp },
    { path: '/recommendations', label: 'Recommendations', icon: List },
    { path: '/twins', label: 'Digital Twins', icon: Cpu },
    { path: '/performance', label: 'Performance', icon: BarChart3 },
  ];

  return (
    <nav className="bg-gray-900 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <TrendingUp className="h-8 w-8 mr-2" />
              <span className="text-xl font-bold">Swing Trading</span>
            </div>
            <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
              {navItems.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium ${
                      isActive
                        ? 'border-blue-500 text-blue-400'
                        : 'border-transparent text-gray-300 hover:text-gray-100 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="h-4 w-4 mr-2" />
                    {item.label}
                  </Link>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}


