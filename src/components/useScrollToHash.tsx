import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const useScrollToHash = () => {
  // Anchor links setting (<a href="#section1">)
  const location = useLocation()
  useEffect(() => {
    if (location.hash) {
      const id = location.hash.substring(1) // Remove the '#' from the hash
      const element = document.getElementById(id)
      // Scroll to element and then adjust for the 60px fixed header
      if (element) element.scrollIntoView()
    } 
    else window.scrollTo({ top: 0})
  }, [location]);
};

export default useScrollToHash;