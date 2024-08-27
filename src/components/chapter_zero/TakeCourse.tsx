import chapterZero from '../../data/chapterZero'
import BaseGrid from '../BaseGrid'

const TakeCourse = () => {
  const [ taking ] = chapterZero[7].sections

  return (
    <BaseGrid section={taking}/>
  )
}

export default TakeCourse