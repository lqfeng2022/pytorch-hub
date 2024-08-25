import chapterOne from '../../data/chapterOne'
import BaseGrid from '../BaseGrid'

const TakeCourse = () => {
  const [ taking ] = chapterOne[7].sections

  return (
    <BaseGrid section={taking}/>
  )
}

export default TakeCourse