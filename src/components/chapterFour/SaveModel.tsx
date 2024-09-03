import Definition from '../Definition'
import CodeDisplay from '../CodeDisplay'
import BaseGrid from '../BaseGrid'
import chapterFour from '../../data/chapterFour'
import codeBinaryModel from '../../data/codeBinaryModel'

const SaveModel = () => {
  const [ define, choose, save, load ] = chapterFour[7].sections
  const [ code_choose, code_save, code_load 
  ] = codeBinaryModel.slice(16, 19).map(obj => obj.code)

  return (
    <div>
      <Definition title={define.name} definition={define}/>
      <BaseGrid section={choose}/>
      <CodeDisplay codes={code_choose}/>
      <BaseGrid section={save}/>
      <CodeDisplay codes={code_save}/>
      <BaseGrid section={load}/>
      <CodeDisplay codes={code_load}/>
    </div>
  )
}

export default SaveModel