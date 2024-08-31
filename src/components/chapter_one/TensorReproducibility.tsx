import BaseGrid from '../BaseGrid'
import Definition from '../Definition'
import chapterOne from '../../data/chapterOne'
import CodeDisplay from '../CodeDisplay'
import pythonCode from '../../data/pythonCode'

const TensorReproducibility = () => {
  const [ 
    repro, rand, randpy, randf, pseudorand, pseudorandf, randseed, randseedpy
  ] = chapterOne[6].sections
  const [randomCode, reproCode] = pythonCode.slice(15, 17).map(obj => obj.code)
  
  return (
    <>
      <Definition title={repro.name} definition={repro}/>
      <Definition title={rand.name} definition={rand}/>
      <BaseGrid section={randpy}/>
      <CodeDisplay codes={randomCode}/>
      <BaseGrid section={randf}/>
      <Definition title={pseudorand.name} definition={pseudorand}/>
      <BaseGrid section={pseudorandf}/>
      <Definition title={randseed.name} definition={randseed}/>
      <BaseGrid section={randseedpy}/>
      <CodeDisplay codes={reproCode}/>
    </>
  )
}

export default TensorReproducibility