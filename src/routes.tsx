import { createBrowserRouter } from 'react-router-dom'
import Layout from './pages/Layout'
import HomePage from './HomePage'
import ChapterZero from './pages/ChapterZero'
import ErrorPage from './pages/ErrorPage'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout/>,
    errorElement: <ErrorPage/>,
    children: [
      { index: true, element: <HomePage/> },
      { path: 'artificial-intelligence', element: <ChapterZero/> }
    ]
  }
])

export default router