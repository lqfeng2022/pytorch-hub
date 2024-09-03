import Definition from '../Definition'
import chapterTwo from '../../data/chapterTwo'
import codeLineModel from '../../data/codeLineModel'
import CodeDisplay from '../CodeDisplay'
import BaseGrid from '../BaseGrid'

const SaveModel = () => {
  const [ define, save, load ] = chapterTwo[4].sections
  const [ savecode, loadcode ] = codeLineModel.slice(9, 11).map(obj => obj.code)

  return (
    <div>
      <Definition title={define.name} definition={define}/>
      <BaseGrid section={save}/>
      <CodeDisplay codes={savecode}/>
      <BaseGrid section={load}/>
      <CodeDisplay codes={loadcode}/>
    </div>
  )
}

export default SaveModel