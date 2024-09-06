import { createBrowserRouter } from 'react-router-dom'
import Layout from './pages/Layout'
import HomePage from './HomePage'
import ChapterZero from './pages/ChapterZero'
import ErrorPage from './pages/ErrorPage'
import Introduction from './pages/Introduction'
import Reference from './pages/Reference'
import AboutMe from './pages/AboutMe'
import ChapterOne from './pages/ChapterOne'
import ChapterTwo from './pages/ChapterTwo'
import ChapterThree from './pages/ChapterThree'
import ChapterFour from './pages/ChapterFour'
import ChapterFive from './pages/ChapterFive'
import ChapterSix from './pages/ChapterSix'
import ChapterSeven from './pages/ChapterSeven'
import ChapterEight from './pages/ChapterEight'
import ChapterNine from './pages/ChapterNine'
import ChapterTen from './pages/ChapterTen'
import ChapterEleven from './pages/ChapterEleven'

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout/>,
    errorElement: <ErrorPage/>,
    children: [
      { index: true, element: <HomePage/> },
      { path: 'introduction', element: <Introduction/> },
      { path: 'artificial-intelligence', element: <ChapterZero/> },
      { path: 'tensors', element: <ChapterOne/> },
      { path: 'a-straight-line-model', element: <ChapterTwo/> },
      { path: 'the-maths-behind-one', element: <ChapterThree/> },
      { path: 'a-binary-classification-model', element: <ChapterFour/> },
      { path: 'the-maths-behind-two', element: <ChapterFive/> },
      { path: 'a-cnn-model', element: <ChapterSix/> },
      { path: 'the-maths-behind-three', element: <ChapterSeven/> },
      { path: 'a-vit-model', element: <ChapterEight/> },
      { path: 'the-maths-behind-four', element: <ChapterNine/> },
      { path: 'a-language-translation-model', element: <ChapterTen/> },
      { path: 'the-maths-behind-five', element: <ChapterEleven/> },
      { path: 'reference', element: <Reference/> },
      { path: 'about-me', element: <AboutMe/> },
    ]
  }
])

export default router