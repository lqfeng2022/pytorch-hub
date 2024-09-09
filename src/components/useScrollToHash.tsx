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
      if (element) {
        setTimeout(() => {
          const elementPosition = element.getBoundingClientRect().top + window.scrollY;
          const offsetPosition = elementPosition - 60; // Adjust for 60px fixed header
          window.scrollTo({ top: offsetPosition })
        }, 100) // Delay for layout to stabilize
      }
    } 
    else window.scrollTo({ top: 0})
  }, [location]);
};

export default useScrollToHash;