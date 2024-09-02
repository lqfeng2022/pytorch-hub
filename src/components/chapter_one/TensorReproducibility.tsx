import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'
import RightGrid from '../RightGrid'
import LeftGrid from '../LeftGrid'

const TensorReproducibility = () => {
  const [ 
    repro, rand, randpy, randf, pseudorand, pseudorandf, randseed, randseedpy
  ] = chapterOne[6].sections
  const [randomCode, reproCode] = pythonCode.slice(20, 22).map(obj => obj.code)
  
  return (
    <>
      <Definition title={repro.name} definition={repro}/>
      <Definition title={rand.name} definition={rand}/>
      <RightGrid section={randpy}/>
      <CodeDisplay codes={randomCode}/>
      <BaseGrid section={randf}/>
      <Definition title={pseudorand.name} definition={pseudorand}/>
      <BaseGrid section={pseudorandf}/>
      <Definition title={randseed.name} definition={randseed}/>
      <LeftGrid section={randseedpy}/>
      <CodeDisplay codes={reproCode}/>
    </>
  )
}

export default TensorReproducibility