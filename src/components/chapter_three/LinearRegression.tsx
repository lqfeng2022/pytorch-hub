import Definition from '../Definition'
import chapterThree from '../../data/chapterThree'
import LeftGrid from '../LeftGrid'

const LinearRegression = () => {
  const [ linear, relation ] = chapterThree[0].sections

  return (
    <>
      <Definition title={linear.name} definition={linear}/>
      <LeftGrid section={relation}/>
    </>
  )
}

export default LinearRegression