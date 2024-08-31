import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import chapterZero from '../../data/chapterZero';

const PTSection = () => {
  const [ whats, features, trends, companies ] = chapterZero[5].sections

  return (
    <>
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={features}/>
      <BaseGrid section={trends}/>
      <BaseGrid section={companies}/>
    </>
  )
}

export default PTSection