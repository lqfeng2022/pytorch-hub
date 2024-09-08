import { createBrowserRouter } from 'react-router-dom'
import HomePage from './HomePage'
import {
  AboutShape,
  ChapterEight,
  ChapterEleven,
  ChapterFive,
  ChapterFour,
  ChapterNine,
  ChapterOne,
  ChapterSeven,
  ChapterSix,
  ChapterTen,
  ChapterThree,
  ChapterTwo,
  ChapterZero,
  ErrorPage,
  Introduction,
  Reference } from './pages'
import Layout from './pages/Layout'

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
      { path: 'a-line-model', element: <ChapterTwo/> },
      { path: 'the-maths-behind-one', element: <ChapterThree/> },
      { path: 'a-classification-model', element: <ChapterFour/> },
      { path: 'the-maths-behind-two', element: <ChapterFive/> },
      { path: 'a-cnn-model', element: <ChapterSix/> },
      { path: 'the-maths-behind-three', element: <ChapterSeven/> },
      { path: 'a-vit-model', element: <ChapterEight/> },
      { path: 'the-maths-behind-four', element: <ChapterNine/> },
      { path: 'a-translation-model', element: <ChapterTen/> },
      { path: 'the-maths-behind-five', element: <ChapterEleven/> },
      { path: 'reference', element: <Reference/> },
      { path: 'about-shape', element: <AboutShape/> },
    ]
  }
])

export default router