import chapterTwo from '../../data/chapterTwo'
import BaseGrid from '../BaseGrid'
import Definition from '../Definition'

const Workflow = () => {
  const [ whats, overview, explain ] =  chapterTwo[0].sections

  return (
    <>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={overview}/>
      <BaseGrid section={explain}/>
    </>
  )
}

export default Workflow