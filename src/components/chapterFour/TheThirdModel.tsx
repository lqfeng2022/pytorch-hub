import { Box } from '@chakra-ui/react'
import chapterFour from '../../data/chapterFour'
import codeBinaryModel from '../../data/codeBinaryModel'
import BaseGrid from '../BaseGrid'
import CodeDisplay from '../CodeDisplay'
import LeftGrid from '../LeftGrid'
import RightGrid from '../RightGrid'

const TheThirdModel = () => {
  const [ three, three_build, three_visual, three_loss ] = chapterFour[6].sections
  const [ three_build_code, three_train_code, three_loss_code
  ] = codeBinaryModel.slice(13, 16).map(obj => obj.code)
  
  return (
    <Box pt={5}>
      <BaseGrid section={three}/>
      <RightGrid section={three_build}/>
      <CodeDisplay codes={three_build_code}/>
      <LeftGrid section={three_visual}/>
      <CodeDisplay codes={three_train_code}/>
      <RightGrid section={three_loss}/>
      <CodeDisplay codes={three_loss_code}/>
    </Box>
  )
}

export default TheThirdModel