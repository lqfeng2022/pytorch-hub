import { Box, Text } from '@chakra-ui/react';
import BaseGrid from '../BaseGrid';
import Definition from '../Definition';
import chapterZero from '../../data/chapterZero';

const PTSection = () => {
  const [ definition, features, trends, companies ] = chapterZero[5].sections

  return (
    <>
      <Box py={5}>
        <Text as='b' fontSize='xl' color='gray.600'>6. PyTorch</Text>
        <Definition definition={definition}/>
      </Box>
      <BaseGrid section={features}/>
      <BaseGrid section={trends}/>
      <BaseGrid section={companies}/>
    </>
  )
}

export default PTSection