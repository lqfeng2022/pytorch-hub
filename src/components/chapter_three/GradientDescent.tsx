import chapterThree from '../../data/chapterThree'
import Definition from '../Definition'
import LeftGrid from '../LeftGrid'
import BaseGrid from '../BaseGrid'

const GradientDescent = () => {
  const [
    whats, gdone, gdone_table, gdone_mse, gdtwo_visual, gdtwo_table, gdtwo 
  ] = chapterThree[3].sections

  return (
    <>  
      <Definition title={whats.name} definition={whats}/>
      <BaseGrid section={gdone}/>
      <LeftGrid section={gdone_table}/>
      <BaseGrid section={gdone_mse}/>
      <BaseGrid section={gdtwo_visual}/>
      <BaseGrid section={gdtwo_table}/>
      <BaseGrid section={gdtwo}/>
    </>
  )
}

export default GradientDescent