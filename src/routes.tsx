import { createBrowserRouter } from 'react-router-dom'
import Layout from './pages/layout'
import HomePage from './HomePage'
import ChapterZero from './components/ChapterZero'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout/>,
    children: [
      { index: true, element: <HomePage/> },
      { path: 'artificial-intelligence', element: <ChapterZero/> }
    ]
  }
])

export default router