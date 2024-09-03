import Definition from '../Definition'
import chapterThree from '../../data/chapterThree'
import BaseGrid from '../BaseGrid'

const LossCurves = () => {
  const [ 
    whats, mse, mse_model, mse_detail, curve, curves 
  ] = chapterThree[2].sections

  return (
    <div>
      <Definition title={whats.name} definition={whats}/>
      <Definition title={mse.name} definition={mse}/>
      <BaseGrid section={mse_model}/>
      <BaseGrid section={mse_detail}/>
      <Definition title={curve.name} definition={curve}/>
      <BaseGrid section={curves}/>
    </div>
  )
}

export default LossCurves